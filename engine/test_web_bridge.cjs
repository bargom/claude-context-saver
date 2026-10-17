const assert = require('node:assert/strict');
const fs = require('node:fs');
const {spawnSync} = require('node:child_process');
const test = require('node:test');
const source = fs.readFileSync(__dirname + '/web_bridge.js', 'utf8')
  .replace("'__CONTEXT_SAVER_LAUNCHER__'", JSON.stringify('/opt/ccs/bin/context-saver'))
  .replace("'__CONTEXT_SAVER_STATE__'", JSON.stringify('/tmp/ccs-state'));
const make = tools => new Function('tools', source)(tools);

test('captures original before emitting summary and preserves request', async () => {
  const request = {search_query: [{q: 'topic'}]};
  const calls = [];
  const web = make({
    web__run: async actual => { assert.equal(actual, request); calls.push('web'); return 'original'; },
    exec_command: async () => { calls.push('helper'); return {output: JSON.stringify({compressed:true,result:'summary'}),exit_code:0}; },
  });
  assert.equal(await web(request), 'summary');
  assert.deepEqual(calls, ['web', 'helper', 'helper']);
});

test('engine commands go through the launcher', async () => {
  const commands = [];
  const web = make({
    web__run: async () => 'original',
    exec_command: async ({cmd}) => { commands.push(cmd); return {output: JSON.stringify({compressed:true,result:'summary'}), exit_code:0}; },
  });
  assert.equal(await web({search_query: [{q: 'topic'}]}), 'summary');
  assert.match(commands[0], / \| '\/opt\/ccs\/bin\/context-saver' web-adapter --decision-only$/);
  assert.match(commands[1], /^'\/opt\/ccs\/bin\/context-saver' delivered --delivery-token /);
});

test('quoted shell payload preserves substitutions, apostrophes and newlines', async () => {
  const original = "caller ' quote\n$(printf UNSAFE) `printf UNSAFE` $HOME \\ τέλος";
  let captured;
  const web = make({
    web__run: async () => original,
    exec_command: async ({cmd}) => {
      // Execute only the generated printf producer, with a harmless stdin echo
      // consumer; the helper/provider never runs in this test.
      const producer = cmd.split(" | '/opt/ccs/")[0];
      const result = spawnSync('/bin/sh', ['-c', producer], {encoding:'utf8'});
      assert.equal(result.status, 0);
      captured = JSON.parse(result.stdout);
      return {output:'{"compressed":false}',exit_code:0};
    },
  });
  assert.equal(await web({open:[{ref_id:'https://example.com'}]}), original);
  assert.equal(captured.tool_response, original);
});

test('raw bypass does not start helper and rich fallback preserves object identity', async () => {
  const original = {content:[{type:'image',data:'abc'}]};
  let count = 0;
  const web = make({web__run:async()=>original,exec_command:async()=>{count++;return {output:'{"compressed":false}',exit_code:0};}});
  assert.equal(await web({}, {raw:true}), original);
  assert.equal(count, 0);
  assert.equal(await web({}), original);
  assert.equal(count, 1);
});

test('transport error, nonzero status and truncated JSON restore exact original', async () => {
  const original = 'full original';
  for (const response of [new Error('transport'),{output:'{"compressed":true',exit_code:0},{output:'bad',exit_code:7}]) {
    const web=make({web__run:async()=>original,exec_command:async()=>{if(response instanceof Error)throw response;return response;}});
    assert.equal(await web({}), original);
  }
});

test('polls yielded helper and joins output chunks', async () => {
  const polls=[];
  const chunks=['true,', '"result":"summary"}'];
  const web=make({
    web__run:async()=>'original',
    exec_command:async({cmd})=>cmd.includes(' web-adapter ')
      ? {output:'{"compressed":',session_id:42} : {output:'',exit_code:0},
    write_stdin:async args=>{
      polls.push(args);
      const output=chunks.shift();
      return chunks.length ? {output,session_id:42} : {output,exit_code:0};
    },
  });
  assert.equal(await web({}), 'summary');
  // The adapter poll keeps its own output budget and waits for the helper.
  assert.deepEqual(polls, [
    {session_id:42,chars:'',max_output_tokens:5000,yield_time_ms:10000},
    {session_id:42,chars:'',max_output_tokens:5000,yield_time_ms:10000},
  ]);
});

test('large web input uses bounded literal chunks and cleans its staging file', async () => {
  const original = "quoted ' $(printf UNSAFE) `printf UNSAFE` 🧬\n".repeat(4000);
  const staged = '/tmp/ccs-state/web-input.TEST1234';
  let captured = '', chunks = 0, removed = false;
  const web = make({
    web__run: async () => original,
    exec_command: async ({cmd}) => {
      if (cmd.startsWith('umask 077\nmktemp ')) return {output:staged+'\n',exit_code:0};
      if (cmd.startsWith("printf '%s' ")) {
        const producer = cmd.slice(0, cmd.lastIndexOf(' >> '));
        const result = spawnSync('/bin/sh', ['-c', producer], {encoding:'utf8'});
        assert.equal(result.status,0);
        assert.ok(Buffer.byteLength(cmd)<150000);
        captured += result.stdout; chunks++;
        return {output:'',exit_code:0};
      }
      if (cmd.includes(' web-adapter ')) {
        assert.equal(JSON.parse(captured).tool_response,original);
        return {output:'{"compressed":true,"result":"bounded summary"}',exit_code:0};
      }
      if (cmd.startsWith('rm -f -- ')) removed=true;
      return {output:'',exit_code:0};
    },
  });
  assert.equal(await web({open:[{ref_id:'https://example.com'}]}),'bounded summary');
  assert.ok(chunks>1);
  assert.ok(removed);
});

test('failed large-input upload preserves original and removes partial staging file', async () => {
  const original='source '.repeat(10000);
  let removed=false;
  const web=make({web__run:async()=>original,exec_command:async({cmd})=>{
    if(cmd.startsWith('umask')) return {output:'/tmp/ccs-state/web-input.TEST1234\n',exit_code:0};
    if(cmd.startsWith('rm -f')) {removed=true;return {output:'',exit_code:0};}
    return {output:'write failed',exit_code:1};
  }});
  assert.equal(await web({}),original);
  assert.ok(removed);
});

test('yielded staging writes finish before adapter and cleanup', async () => {
  let writing=false, removed=false, polls=0;
  const web=make({
    web__run:async()=> 'large '.repeat(6000),
    exec_command:async({cmd})=>{
      if(cmd.startsWith('umask')) return {output:'/tmp/ccs-state/web-input.',session_id:11};
      if(cmd.startsWith('printf')) {writing=true;return {output:'',session_id:12};}
      if(cmd.includes(' web-adapter ')) {assert.equal(writing,false);return {output:'{"compressed":true,"result":"summary"}',exit_code:0};}
      if(cmd.startsWith('rm -f')) {assert.equal(writing,false);removed=true;}
      return {output:'',exit_code:0};
    },
    write_stdin:async({session_id})=>{
      polls++;
      if(session_id===11) return {output:'TEST1234\n',exit_code:0};
      assert.equal(session_id,12);writing=false;return {output:'',exit_code:0};
    },
  });
  assert.equal(await web({}),'summary');
  assert.ok(polls>=3);
  assert.ok(removed);
});
