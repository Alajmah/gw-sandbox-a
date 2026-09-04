#!/usr/bin/env python3
from __future__ import annotations
import json, os, random, re, time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
CASES = json.loads((ROOT / 'pilot_cases.json').read_text(encoding='utf-8'))
TARGET = os.getenv('OPENAI_TARGET_MODEL', 'gpt-5.6-sol')
JUDGE = os.getenv('OPENAI_JUDGE_MODEL', TARGET)
API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=API_KEY, base_url=os.getenv('OPENAI_BASE_URL') or None)

COMPACT = '''You are a reasoning agent. For non-trivial tasks, use internally: Problem → First Principle → Mechanism → Evidence → Solution. Define the actual problem before solving it. Distinguish observation, inference, assumption, hypothesis, prediction, evidence, conclusion, and recommendation. First principles are necessities/invariants, not conventions or existing implementations. A plausible explanation is not an established cause; correlation is not mechanism. Expose decision-critical assumptions and uncertainty. Seek evidence capable of weakening the favored mechanism. Revise when evidence contradicts the model. Engineer from the best-supported mechanism under constraints, including risk, reversibility, side effects, and second-order effects. Stop when more information is unlikely to change the action. For trivial or established tasks, answer directly. Do not mechanically emit protocol headings.'''

FULL = '''You are a reasoning agent operating under: Observe → Diagnose → Derive → Hypothesize → Predict → Test → Revise → Engineer. For trivial tasks answer directly; for moderate established tasks compress to Problem → First Principle → Mechanism → Evidence → Solution. Separate observations from interpretations. Generate multiple plausible explanations under material ambiguity. Distinguish symptoms, proximate causes, and root causes. Derive first principles as necessities/invariants, not conventions, precedent, analogies, or current implementations. Express important explanations as falsifiable mechanisms; derive observable predictions and prefer discriminating/falsifying tests. Contradictions require model revision. Preserve uncertainty; confidence tracks evidence. Expose critical assumptions. Engineer only after sufficient understanding; compare effectiveness, robustness, cost, risk, reversibility, feasibility, constraints, feedback loops, side effects, and second-order effects. Continue when uncertainty is material and reducible; stop when further information has insufficient decision value. After intervention, predict and observe outcomes. Do not mechanically expose the entire internal state.'''

ROUTER = '''Choose the least expensive reasoning mode that preserves reliability. DIRECT: trivial, deterministic, or established answer where causal investigation adds no value. COMPACT: moderately complex task with a reasonably established mechanism or one decision-critical uncertainty. FULL: substantial ambiguity, competing causal explanations, contradictory/sequential evidence, high stakes, or important second-order/robustness analysis. Return JSON only: {"mode":"DIRECT|COMPACT|FULL","rationale":"one short sentence"}.'''

PROTO_JUDGE = '''You are a blinded evaluator of reasoning-protocol behavior. You are not told which experimental condition produced the response. Score only the dimensions listed as applicable from 0 to 4. Do not reward headings, protocol jargon, or verbosity. Use only the transcript prefix and turn-specific reference; never infer future evidence. Return JSON only: {"scores":{"criterion":0},"violations":["Pxx"],"notes":"brief justification"}. Relevant violation codes: P01 premature solutioning; P02 observation/inference collapse; P03 single-hypothesis fixation; P04 principle laundering; P05 correlation as mechanism; P06 unfalsifiable explanation; P07 post-hoc prediction; P08 confirmation-only testing; P09 evidence-free confidence; P10 model-preservation bias; P11 uncertainty erasure; P12 hidden critical assumption; P13 constraint blindness; P14 solution monoculture; P15 solution-model disconnect; P16 local optimization; P17 fragile decision; P18 premature irreversibility; P19 premature termination; P20 analysis paralysis; P21 ceremonial compliance.'''

OUTCOME_JUDGE = '''You are a blinded evaluator of answer and decision quality. Ignore named reasoning frameworks. Score only listed applicable dimensions from 0 to 4. Use only the transcript prefix and turn-specific reference; never use future evidence. decision_quality means the quality of the proposed decision or next action given information currently available. Return JSON only: {"scores":{"criterion":0},"notes":"brief justification"}.'''

PAIRWISE_JUDGE = '''You are a blinded pairwise evaluator. Compare Response A and Response B for the same case. You are not told which condition produced either. Use the supplied reference as ground truth. Do not reward framework terminology, headings, or verbosity. Prefer better correctness, causal/mechanistic reasoning where relevant, evidence use, revision across turns, calibrated uncertainty, decision quality, robustness, and efficient depth. For sequential cases judge the trajectory against evidence available at each turn. Return JSON only: {"winner":"A|B|TIE","confidence":0.0,"notes":"brief reason"}.'''

CONDITION_PROMPTS = {'BASELINE': None, 'COMPACT': COMPACT, 'FULL': FULL}


def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def call(system, messages, model):
    kwargs = {'model': model, 'input': messages, 'store': False}
    if system:
        kwargs['instructions'] = system
    max_out = os.getenv('OPENAI_MAX_OUTPUT_TOKENS')
    if max_out:
        kwargs['max_output_tokens'] = int(max_out)
    effort = os.getenv('OPENAI_REASONING_EFFORT')
    if effort:
        kwargs['reasoning'] = {'effort': effort}
    last = None
    for attempt in range(4):
        started = time.perf_counter()
        try:
            r = client.responses.create(**kwargs)
            u = r.usage
            return {
                'text': r.output_text or '',
                'usage': {
                    'input_tokens': int(getattr(u, 'input_tokens', 0) or 0),
                    'output_tokens': int(getattr(u, 'output_tokens', 0) or 0),
                    'latency_ms': (time.perf_counter()-started)*1000,
                },
                'response_id': r.id,
            }
        except Exception as e:
            last = e
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise last


def add_usage(total, usage):
    for k in total:
        total[k] += usage.get(k, 0) or 0


def route(case):
    r = call(ROUTER, [{'role':'user','content':case['turns'][0]['agent_input']}], TARGET)
    p = parse_json(r['text'])
    mode = p.get('mode')
    if mode not in {'DIRECT','COMPACT','FULL'}:
        raise RuntimeError(f"invalid router mode: {p}")
    return mode, p.get('rationale',''), r['usage']


def run_case(case, condition):
    usage = {'input_tokens':0,'output_tokens':0,'latency_ms':0.0}
    mode, rationale = None, None
    if condition == 'ADAPTIVE':
        mode, rationale, u = route(case)
        add_usage(usage,u)
        system = None if mode == 'DIRECT' else (COMPACT if mode == 'COMPACT' else FULL)
    else:
        system = CONDITION_PROMPTS[condition]
        mode = condition if condition in {'COMPACT','FULL'} else 'DIRECT'
    messages, responses = [], []
    for t in case['turns']:
        messages.append({'role':'user','content':t['agent_input']})
        out = call(system, messages, TARGET)
        add_usage(usage,out['usage'])
        responses.append({'turn':t['turn'],'text':out['text']})
        messages.append({'role':'assistant','content':out['text']})
    return {'case_id':case['case_id'],'condition':condition,'selected_mode':mode,'router_rationale':rationale,'responses':responses,'usage':usage}


def transcript(case, run, through):
    rs = {r['turn']:r['text'] for r in run['responses']}
    chunks=[]
    for t in case['turns']:
        if t['turn'] > through:
            break
        chunks += [f"TURN {t['turn']} USER:\n{t['agent_input']}", f"TURN {t['turn']} ASSISTANT:\n{rs[t['turn']]}"]
    return '\n\n'.join(chunks)


def judge_turn(case, run, t, kind):
    profile=case['evaluation_profile']
    dims=list(profile['protocol_dimensions'] if kind=='protocol' else profile['outcome_dimensions'])
    dims=[d for d in dims if d!='mode_fit']
    if t['turn']==1:
        dims=[d for d in dims if d!='revision']
    ref=t.get('judge_reference',{}).get(kind,[])
    content={'case_id':case['case_id'],'turn':t['turn'],'transcript_prefix':transcript(case,run,t['turn']),'applicable_dimensions':dims,'turn_reference':ref}
    out=call(PROTO_JUDGE if kind=='protocol' else OUTCOME_JUDGE,[{'role':'user','content':json.dumps(content,ensure_ascii=False)}],JUDGE)
    parsed=parse_json(out['text'])
    scores={k:float(v) for k,v in parsed.get('scores',{}).items() if k in dims and isinstance(v,(int,float)) and 0<=float(v)<=4}
    return {'scores':scores,'violations':parsed.get('violations',[]) if kind=='protocol' else [],'notes':parsed.get('notes',''),'usage':out['usage']}


def evaluate(case, run):
    turns=[]; violations=set(); pacc={}; oacc={}; judge_usage={'input_tokens':0,'output_tokens':0,'latency_ms':0.0}
    for t in case['turns']:
        p=judge_turn(case,run,t,'protocol'); o=judge_turn(case,run,t,'outcome')
        add_usage(judge_usage,p['usage']); add_usage(judge_usage,o['usage'])
        violations.update(p['violations'])
        for k,v in p['scores'].items():
            pacc.setdefault(k,[]).append(v)
        for k,v in o['scores'].items():
            oacc.setdefault(k,[]).append(v)
        turns.append({'turn':t['turn'],'protocol':{k:v for k,v in p.items() if k!='usage'},'outcome':{k:v for k,v in o.items() if k!='usage'}})
    pavg={k:sum(v)/len(v) for k,v in pacc.items()}; oavg={k:sum(v)/len(v) for k,v in oacc.items()}
    return {'protocol_scores':pavg,'outcome_scores':oavg,'protocol_mean':sum(pavg.values())/len(pavg) if pavg else 0,'quality_mean':sum(oavg.values())/len(oavg) if oavg else 0,'violations':sorted(violations),'turns':turns,'judge_usage':judge_usage}


def pairwise(case, base, challenger):
    a,b=base,challenger
    flipped=random.Random(case['case_id']+challenger['condition']).random()<0.5
    if flipped:
        a,b=b,a
    ref={k:v for k,v in case['evaluator_key'].items() if k in {'observations','not_observed','plausible_hypotheses','critical_first_principles','discriminating_evidence','decision_critical_uncertainties','constraints','acceptable_interventions','scoring_notes'} and v}
    last=case['turns'][-1]['turn']
    content={'case_id':case['case_id'],'expected_mode':case['expected_mode'],'reference':ref,'response_A':transcript(case,a,last),'response_B':transcript(case,b,last)}
    out=call(PAIRWISE_JUDGE,[{'role':'user','content':json.dumps(content,ensure_ascii=False)}],JUDGE)
    p=parse_json(out['text']); w=p.get('winner')
    if w not in {'A','B','TIE'}:
        raise RuntimeError(f'invalid pairwise: {p}')
    resolved='TIE' if w=='TIE' else ((a if w=='A' else b)['condition'])
    return {'case_id':case['case_id'],'challenger':challenger['condition'],'winner':resolved,'confidence':p.get('confidence'),'notes':p.get('notes',''),'usage':out['usage']}


def main():
    outdir=ROOT/'results'; outdir.mkdir(exist_ok=True)
    runs=[]
    for case in CASES:
        for cond in ['BASELINE','COMPACT','FULL','ADAPTIVE']:
            print('running',case['case_id'],cond,flush=True)
            r=run_case(case,cond)
            print('evaluating',case['case_id'],cond,flush=True)
            r['evaluation']=evaluate(case,r)
            runs.append(r)
    index={(r['case_id'],r['condition']):r for r in runs}
    pairs=[]
    for case in CASES:
        base=index[(case['case_id'],'BASELINE')]
        for cond in ['COMPACT','FULL','ADAPTIVE']:
            print('pairwise',case['case_id'],cond,flush=True)
            pairs.append(pairwise(case,base,index[(case['case_id'],cond)]))
    summary={}
    for cond in ['BASELINE','COMPACT','FULL','ADAPTIVE']:
        rr=[r for r in runs if r['condition']==cond]
        summary[cond]={
            'n':len(rr),
            'quality_mean':sum(r['evaluation']['quality_mean'] for r in rr)/len(rr),
            'protocol_mean':sum(r['evaluation']['protocol_mean'] for r in rr)/len(rr),
            'input_tokens':sum(r['usage']['input_tokens'] for r in rr),
            'output_tokens':sum(r['usage']['output_tokens'] for r in rr),
            'latency_ms':sum(r['usage']['latency_ms'] for r in rr),
            'pairwise_vs_baseline':{
                'wins':sum(1 for p in pairs if p['challenger']==cond and p['winner']==cond),
                'ties':sum(1 for p in pairs if p['challenger']==cond and p['winner']=='TIE'),
                'losses':sum(1 for p in pairs if p['challenger']==cond and p['winner']=='BASELINE'),
            } if cond!='BASELINE' else None,
        }
    report={'target_model':TARGET,'judge_model':JUDGE,'note':'Pilot replicate labels are stochastic because the current Responses API exposes no seed parameter.','summary':summary,'runs':runs,'pairwise':pairs}
    (outdir/'pilot_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (outdir/'pilot_runs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in runs),encoding='utf-8')
    (outdir/'pilot_pairwise.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pairs),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
