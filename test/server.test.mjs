import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CURATED_MODELS,
  handleChat,
  handleModels,
  OPENCODE_FREE_MODELS,
  learningRecordsFromState,
  learningStateFromRecords,
  normalizeProviderModels,
  parseEvaluationResponse,
  validateEvaluationContext,
  validateLearningPayload,
} from '../scripts/server.mjs';

function responseRecorder() {
  return {
    status: null,
    headers: null,
    body: '',
    writeHead(status, headers) {
      this.status = status;
      this.headers = headers;
    },
    end(body) {
      this.body = body;
    },
  };
}

test('chat forwards provider rate limits with a 70 second retry window', async () => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.GROQ_API_KEY;
  process.env.GROQ_API_KEY = 'test-key';
  globalThis.fetch = async () => new Response(JSON.stringify({
    error: { message: 'rate limit' },
  }), {
    status: 429,
    headers: { 'Content-Type': 'application/json' },
  });

  try {
    const response = responseRecorder();
    await handleChat({
      method: 'POST',
      body: {
        provider: 'groq',
        model: 'qwen/qwen3.6-27b',
        harness: 'normal',
        messages: [{ role: 'user', content: 'hola' }],
      },
    }, response);

    assert.equal(response.status, 429);
    assert.equal(response.headers['Retry-After'], '70');
    assert.deepEqual(JSON.parse(response.body), {
      error: 'rate limit',
      retryAfterSeconds: 70,
    });
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.GROQ_API_KEY;
    else process.env.GROQ_API_KEY = originalKey;
  }
});

test('chat also forwards a plain-text provider rate limit', async () => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.GROQ_API_KEY;
  process.env.GROQ_API_KEY = 'test-key';
  globalThis.fetch = async () => new Response('try again later', { status: 429 });

  try {
    const response = responseRecorder();
    await handleChat({
      method: 'POST',
      body: {
        provider: 'groq',
        model: 'qwen/qwen3.6-27b',
        harness: 'normal',
        messages: [{ role: 'user', content: 'hola' }],
      },
    }, response);
    assert.equal(response.status, 429);
    assert.equal(JSON.parse(response.body).retryAfterSeconds, 70);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.GROQ_API_KEY;
    else process.env.GROQ_API_KEY = originalKey;
  }
});

test('OpenCode Zen sends curated free models to its OpenAI-compatible endpoint', async () => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.OPENCODE_API_KEY;
  let request;
  process.env.OPENCODE_API_KEY = 'test-key';
  globalThis.fetch = async (url, options) => {
    request = { url, options };
    return new Response(JSON.stringify({
      choices: [{ message: { content: 'respuesta' }, finish_reason: 'stop' }],
    }), { headers: { 'Content-Type': 'application/json' } });
  };

  try {
    const response = responseRecorder();
    await handleChat({
      method: 'POST',
      body: {
        provider: 'opencode',
        model: 'mimo-v2.5-free',
        harness: 'normal',
        messages: [{ role: 'user', content: 'hola' }],
      },
    }, response);

    assert.equal(response.status, 200);
    assert.equal(request.url, 'https://opencode.ai/zen/v1/chat/completions');
    assert.equal(request.options.headers.Authorization, 'Bearer test-key');
    assert.equal(JSON.parse(request.options.body).model, 'mimo-v2.5-free');
    assert.equal(JSON.parse(response.body).content, 'respuesta');
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.OPENCODE_API_KEY;
    else process.env.OPENCODE_API_KEY = originalKey;
  }
});

test('chat rejects models outside the curated catalog', async () => {
  const originalKey = process.env.OPENCODE_API_KEY;
  process.env.OPENCODE_API_KEY = 'test-key';

  try {
    assert.equal(OPENCODE_FREE_MODELS.has('big-pickle'), true);
    const response = responseRecorder();
    await handleChat({
      method: 'POST',
      body: {
        provider: 'opencode',
        model: 'big-pickle',
        harness: 'normal',
        messages: [{ role: 'user', content: 'hola' }],
      },
    }, response);
    assert.equal(response.status, 400);
    assert.match(JSON.parse(response.body).error, /no permitido/);
  } finally {
    if (originalKey === undefined) delete process.env.OPENCODE_API_KEY;
    else process.env.OPENCODE_API_KEY = originalKey;
  }
});

test('models endpoint returns the fixed curated catalog in order', async () => {
  const response = responseRecorder();
  await handleModels({ method: 'GET' }, response);

  assert.equal(response.status, 200);
  assert.deepEqual(JSON.parse(response.body).providers,
    Object.entries(CURATED_MODELS).map(([id, models]) => ({ id, models })));
});

test('model catalogs keep only free and chat-compatible entries', () => {
  assert.deepEqual(normalizeProviderModels('openrouter', { data: [
    { id: 'free/chat:free', pricing: { prompt: '0', completion: '0' } },
    { id: 'paid/chat', pricing: { prompt: '0.1', completion: '0.2' } },
    { id: 'free/safety:free', pricing: { prompt: '0', completion: '0' } },
  ] }), ['free/chat:free']);

  assert.deepEqual(normalizeProviderModels('opencode', { data: [
    { id: 'big-pickle' }, { id: 'gpt-5.6-sol' },
  ] }), ['big-pickle']);

  assert.deepEqual(normalizeProviderModels('groq', { data: [
    { id: 'chat-model' }, { id: 'allam-2-7b' }, { id: 'whisper-large-v3' },
    { id: 'prompt-guard' }, { id: 'orpheus-voice' },
  ] }), ['chat-model']);

  assert.deepEqual(normalizeProviderModels('mistral', { data: [
    { id: 'mistral-small-latest', capabilities: { completion_chat: true } },
    { id: 'mistral-medium-2604', capabilities: { completion_chat: true } },
    { id: 'mistral-medium-2505', capabilities: { completion_chat: true } },
    { id: 'mistral-vibe-cli-latest', capabilities: { completion_chat: true } },
    { id: 'embed', capabilities: { completion_chat: false } },
  ] }), ['mistral-small-latest']);

  assert.deepEqual(normalizeProviderModels('cohere', { models: [
    { name: 'north-mini-code-1-0', endpoints: ['chat'], is_deprecated: false },
    { name: 'tiny-aya-earth', endpoints: ['chat'], is_deprecated: false },
    { name: 'old', endpoints: ['chat'], is_deprecated: true },
  ] }), ['north-mini-code-1-0']);

  assert.deepEqual(normalizeProviderModels('google', { models: [
    { name: 'models/gemini-pro-latest', supportedGenerationMethods: ['generateContent'] },
    { name: 'models/gemini-flash-latest', supportedGenerationMethods: ['generateContent'] },
    { name: 'models/gemini-pro-preview', supportedGenerationMethods: ['generateContent'] },
    { name: 'models/embed', supportedGenerationMethods: ['embedContent'] },
  ] }), ['gemini-flash-latest', 'gemini-pro-latest']);
});

test('learning payload metadata must match its state', () => {
  const state = {
    schemaVersion: 1,
    settings: {
      trackId: 'track',
      trackVersion: 1,
      selectedLevelId: 'nivel-0',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    lessonRuns: [],
    skillProgress: {},
    reviewQueue: [],
    currentRunId: null,
  };
  assert.doesNotThrow(() => validateLearningPayload({
    revision: 0,
    trackId: 'track',
    trackVersion: 1,
    selectedLevelId: 'nivel-0',
    state,
  }));
  assert.throws(() => validateLearningPayload({
    revision: 0,
    trackId: 'other',
    trackVersion: 1,
    selectedLevelId: 'nivel-0',
    state,
  }), /no coinciden/);
});

test('learning state is stored as independent records and rebuilt without deleted runs', () => {
  const settings = {
    trackId: 'track',
    trackVersion: 1,
    selectedLevelId: 'nivel-0',
    createdAt: '2026-01-01T00:00:00.000Z',
    updatedAt: '2026-01-01T00:00:00.000Z',
  };
  const run = { id: 'run-1', lessonId: 'lesson-1', status: 'in_progress' };
  const review = { id: 'review-1', lessonId: 'lesson-1', status: 'pending' };
  const state = {
    schemaVersion: 1,
    settings,
    lessonRuns: [run],
    skillProgress: { arrays: { score: 80 } },
    reviewQueue: [review],
    currentRunId: run.id,
  };
  const records = learningRecordsFromState(state);

  assert.deepEqual(records.map(record => record.record_type), ['run', 'skill', 'review']);
  assert.deepEqual(learningStateFromRecords({
    schema_version: 1,
    current_run_id: run.id,
    settings,
  }, records), state);

  const withoutRun = learningStateFromRecords({
    schema_version: 1,
    current_run_id: run.id,
    settings,
  }, records.filter(record => record.record_type !== 'run'));
  assert.deepEqual(withoutRun.lessonRuns, []);
  assert.equal(withoutRun.currentRunId, null);
});

test('evaluation evidence is restricted to the lesson skills', () => {
  const raw = JSON.stringify({
    verdict: 'passed',
    score: 90,
    criticalChecksPassed: true,
    feedback: 'Correcto.',
    skillEvidence: [{ skillId: 'filter-simple', score: 90 }],
    nextAction: 'complete',
  });
  assert.equal(parseEvaluationResponse(raw, 80, ['filter-simple']).score, 90);
  assert.throws(() => parseEvaluationResponse(raw, 80, ['otra-habilidad']), /no coincide/);
});

test('evaluation context supports project files and keeps console payloads compatible', () => {
  const base = {
    lessonId: 'lesson',
    title: 'Lección',
    type: 'lesson',
    task: { instructions: ['Completa el ejercicio.'] },
    dataset: null,
    acceptanceCriteria: ['Cumple el resultado.'],
    rubric: [{ id: 'resultado', points: 100 }],
    criticalChecks: [{ id: 'completo', description: 'Está completo.' }],
    reference: { executableCode: 'console.log(true);' },
    submission: 'console.log(true);',
    skills: ['codigo'],
  };

  const consoleContext = validateEvaluationContext(base);
  assert.equal(consoleContext.modality, 'console');
  assert.deepEqual(consoleContext.submissionFiles, []);

  const projectContext = validateEvaluationContext({
    ...base,
    modality: 'project_files',
    runtime: { environment: 'browser', framework: 'react' },
    workspace: { projectId: 'vite-react', url: 'http://localhost:5173' },
    submissionFiles: ['src/App.jsx', 'src/App.css'],
    expectedBrowserResult: ['Muestra el resultado solicitado.'],
    reference: {
      files: [
        { path: 'src/App.jsx', content: 'export default function App() {}' },
        { path: 'src/App.css', content: 'main {}' },
      ],
    },
    submission: 'export default function App() {}\n\nmain {}',
  });
  assert.equal(projectContext.modality, 'project_files');
  assert.deepEqual(projectContext.submissionFiles, ['src/App.jsx', 'src/App.css']);

  assert.throws(
    () => validateEvaluationContext({ ...base, modality: 'project_files' }),
    /requiere runtime y workspace/,
  );
});

test('project evaluation prompt accepts ordered files without mandatory headers', async () => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.GROQ_API_KEY;
  let providerBody;
  process.env.GROQ_API_KEY = 'test-key';
  globalThis.fetch = async (_url, options) => {
    providerBody = JSON.parse(options.body);
    return new Response(JSON.stringify({
      choices: [{
        message: {
          content: JSON.stringify({
            verdict: 'passed',
            score: 90,
            criticalChecksPassed: true,
            feedback: 'Los archivos cumplen la consigna.',
            skillEvidence: [{ skillId: 'react-state', score: 90 }],
            nextAction: 'complete',
          }),
        },
        finish_reason: 'stop',
      }],
      usage: {},
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };

  try {
    const response = responseRecorder();
    await handleChat({
      method: 'POST',
      body: {
        provider: 'groq',
        model: 'qwen/qwen3.6-27b',
        intent: 'lesson_evaluate',
        evaluationContext: {
          lessonId: 'react-project',
          title: 'Estado en React',
          type: 'lesson',
          modality: 'project_files',
          skills: ['react-state'],
          task: { instructions: ['Completa App y Counter.'] },
          dataset: null,
          runtime: { environment: 'browser', framework: 'react' },
          workspace: { projectId: 'vite-react', url: 'http://localhost:5173' },
          submissionFiles: ['src/App.jsx', 'src/Counter.jsx'],
          expectedBrowserResult: ['El contador aumenta.'],
          acceptanceCriteria: ['Usa estado local.'],
          rubric: [{ id: 'resultado', points: 100 }],
          criticalChecks: [{ id: 'estado', description: 'El estado vive en App.' }],
          reference: { files: [{ path: 'src/App.jsx', content: 'referencia' }] },
          submission: 'export default function App() {}\n\nexport default function Counter() {}',
        },
      },
    }, response);

    assert.equal(response.status, 200);
    assert.match(providerBody.messages[0].content, /encabezados de ruta son opcionales/);
    assert.match(providerBody.messages[0].content, /no afirmes que ejecutaste el proyecto/);
    assert.match(providerBody.messages[1].content, /"modality":"project_files"/);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.GROQ_API_KEY;
    else process.env.GROQ_API_KEY = originalKey;
  }
});
