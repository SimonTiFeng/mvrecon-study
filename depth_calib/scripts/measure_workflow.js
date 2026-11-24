export const meta = {
  name: 'measure-batch',
  description: 'Claude vision agents measure images through the frozen MCP submission contract',
  phases: [{ title: 'Measure' }],
}

const P = 'C:/Users/Simon/Downloads/Git/research/music/depth_calib'
const runDir = args.run_dir
const mode = args.mode || 'harness'
const toolset = args.toolset || 'all'
const targets = args.targets || {}
const samples = args.samples

const SAMPLE_SCHEMA = {
  type: 'object',
  properties: {
    sample_id: { type: 'string' },
    accepted: { type: 'boolean' },
    anchor_count: { type: 'integer' },
    summary: { type: 'string' },
  },
  required: ['sample_id', 'accepted', 'anchor_count', 'summary'],
  additionalProperties: false,
}

function toolSection(sid) {
  const ctx = `${P}/results/${runDir}/samples/${sid}/sample_context.json`
  const py = `${P}/.venv/Scripts/python.exe`
  const client = `${P}/src/frontends/agent_client.py`
  if (toolset === 'none') {
    return `本模式不提供任何局部查看或候选点工具，只能直接提交：
  ${py} ${client} submit --context ${ctx} --payload <payload.json>`
  }
  if (toolset === 'points') {
    return `工具（在 Bash 中调用；服务端计费：最多 4 次观察调用，其中 crop 最多 2 次）：
  ${py} ${client} crop   --context ${ctx} --bbox "x0 y0 x1 y1"
  ${py} ${client} points --context ${ctx} --roi "x0 y0 x1 y1" --max 16
  ${py} ${client} render --context ${ctx} --payload <pairs.json>
  ${py} ${client} submit --context ${ctx} --payload <payload.json>
本条件是"候选点组"：最终 p1/p2 必须使用 points 返回的 point_id（xy=null），不允许自填坐标或编造 ID。
crop/points 的 JSON 结果里有 view_path，用 Read 查看该 PNG 后再判断。`
  }
  return `工具（在 Bash 中调用；服务端计费：最多 4 次观察调用，其中 crop 最多 2 次）：
  ${py} ${client} crop   --context ${ctx} --bbox "x0 y0 x1 y1"
  ${py} ${client} render --context ${ctx} --payload <pairs.json>
  ${py} ${client} submit --context ${ctx} --payload <payload.json>
本条件是"直接指点组"：没有候选点工具，p1/p2 必须直接填写原图像素坐标 [x,y]（可多次修改后再提交）。
crop 的 JSON 结果里有 view_path，用 Read 查看该 PNG 后再判断。`
}

function buildPrompt(sample) {
  const sid = sample.sample_id
  const W = sample.w, H = sample.h
  const img = `${P}/results/${runDir}/samples/${sid}/input.png`
  const tgt = targets[sid]
  const targetLine = tgt
    ? `\n本样本是固定目标对照：请只测量【${tgt}】，两个端点都落在该目标上，不要改测其他物体。\n`
    : ''
  return `你是本实验的视觉测量助手，只处理样本 ${sid}。严格按协议完成，不要做别的事。

样本：image_id = ${sid}，原图尺寸 W=${W}, H=${H}，坐标原点左上角，x 向右、y 向下，单位是原图像素中心坐标。
原图路径（用 Read 工具查看）：${img}
${targetLine}
${toolSection(sid)}

规则：
- 自主寻找具有可用世界知识的物理长度参照（具体型号、标准规格、类别先验均可）。不得搜索网页或读取外部答案。
- 每条证据必须是两个可见表面点之间的三维直线距离，单位米；不得用像素长度或相机距离代替。
- 给出简短 visible_evidence 与 identity_uncertainty；优先独立可靠证据，不为凑数量猜测。
- 没有可靠证据时提交 status=abstain、anchors=[]、非空 abstain_reason。
- 严禁读取真值深度、GT 文件、manifest、评测分数或其他方法结果；只允许查看本样本的 input.png 与工具返回的 view_path 图片。不要修改任何项目文件。

提交（把 JSON 写到 %TEMP% 下的临时文件，然后 submit）：
{"image_id":"${sid}","status":"ok","anchors":[{"anchor_id":"a1","object_group_id":"...","knowledge_group_id":"...","description":"...","p1":{...},"p2":{...},"length_m":1.0,"length_interval_m":[lo,hi]或null,"prior_kind":"identified_standard|identified_product|category_prior|other","visible_evidence":"...","identity_uncertainty":"..."}],"abstain_reason":null}
anchors 限 1-6 条。accepted=true 后立即结束；若返回 violations，可按提示修正一次再提交。
完成后返回结构化结果：sample_id、accepted、anchor_count、summary（每条 anchor 一句话，或 abstain 原因）。`
}

phase('Measure')
const results = await pipeline(
  samples,
  sample => agent(buildPrompt(sample), { schema: SAMPLE_SCHEMA, phase: 'Measure', label: `measure:${sample.sample_id}` }),
)
return { measured: results.filter(Boolean), missing: results.filter(r => !r).length }
