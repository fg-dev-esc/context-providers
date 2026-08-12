import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import test from 'node:test';

const fixtureUrl = new URL('./fixtures/synthetic/conversation.json', import.meta.url);
const rootUrl = new URL('../', import.meta.url);

test('portfolio files and synthetic conversation fixture are valid', async () => {
  await access(new URL('ARCHITECTURE.md', rootUrl));
  await access(new URL('legacy/README.md', rootUrl));

  const fixture = JSON.parse(await readFile(fixtureUrl, 'utf8'));
  assert.equal(fixture.synthetic, true);
  assert.equal(fixture.conversation.messages.length, 2);
  assert.equal(fixture.conversation.messages[0].role, 'user');
  assert.equal(fixture.conversation.messages[1].role, 'assistant');
});
