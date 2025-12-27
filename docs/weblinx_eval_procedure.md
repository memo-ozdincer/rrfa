# WebLINX Custom Evaluation Procedure

## Overview

WebLINX requires custom evaluation for RLVR training since it has ground truth actions but no built-in automated evaluation. This document describes how to implement a custom eval that measures:

1. **Element selection accuracy** (40% weight): Did agent pick correct UID from candidates?
2. **Action type accuracy** (40% weight): Did agent use correct action type?
3. **Dialogue quality** (20% weight): Semantic similarity of utterances (for "say" actions)

## Implementation Status

**⚠️ NOT YET IMPLEMENTED - INFRASTRUCTURE ONLY**

This procedure should be implemented when RLVR training begins. For now, the data formatting scripts skip custom eval and focus on getting data into the right format.

---

## Evaluation Function

### Core Algorithm

```python
from sentence_transformers import SentenceTransformer, util

# Load semantic similarity model (once, globally)
SIMILARITY_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_weblinx_action(agent_action: dict, ground_truth: dict) -> float:
    """
    Evaluate agent action against ground truth.

    Args:
        agent_action: {
            'type': 'click' | 'textInput' | 'load' | 'say' | ...,
            'uid': '67e2a5fb-8b1d-41a0',  # for element actions
            'text': '...',                 # for textInput
            'utterance': '...',            # for say
            'url': '...',                  # for load
        }
        ground_truth: Same schema as agent_action

    Returns:
        Score between 0.0 and 1.0
    """
    score = 0.0

    # 1. Element Selection (40% weight)
    # Check if agent picked correct element from candidates
    if 'uid' in ground_truth:
        if agent_action.get('uid') == ground_truth['uid']:
            score += 0.4
        # Partial credit if UID is close (same element type, nearby in DOM)
        elif is_similar_element(agent_action.get('uid'), ground_truth['uid']):
            score += 0.2

    # 2. Action Type (40% weight)
    # Check if agent used correct action type
    if agent_action.get('type') == ground_truth['type']:
        score += 0.4

    # 3. Dialogue Quality (20% weight)
    # Only for 'say' actions - measure semantic similarity
    if ground_truth['type'] == 'say':
        agent_utt = agent_action.get('utterance', '')
        gt_utt = ground_truth['utterance']

        if agent_utt and gt_utt:
            # Compute cosine similarity using sentence-transformers
            emb1 = SIMILARITY_MODEL.encode(agent_utt)
            emb2 = SIMILARITY_MODEL.encode(gt_utt)
            similarity = util.cos_sim(emb1, emb2).item()

            # Normalize to 0-1 (cosine sim is already -1 to 1, but usually 0-1 for similar text)
            score += 0.2 * max(0, similarity)

    return score


def is_similar_element(uid1: str, uid2: str, candidates: list) -> bool:
    """
    Check if two UIDs refer to similar elements (same tag, nearby in DOM).

    This provides partial credit for "close enough" selections.
    """
    # Parse candidates to find elements
    elem1 = find_element_by_uid(uid1, candidates)
    elem2 = find_element_by_uid(uid2, candidates)

    if not (elem1 and elem2):
        return False

    # Same tag type?
    if elem1.get('tag') == elem2.get('tag'):
        # Check XPath similarity (nearby in DOM tree)
        xpath1 = elem1.get('xpath', '')
        xpath2 = elem2.get('xpath', '')

        # Simple heuristic: same depth in tree and similar path
        if xpath_similarity(xpath1, xpath2) > 0.7:
            return True

    return False


def find_element_by_uid(uid: str, candidates: str) -> dict:
    """
    Parse candidates string to find element with given UID.

    Candidates format:
    "(uid = 67e2a5fb...) [[tag]] input [[xpath]] /html/body/... [[text]] Submit"
    """
    import re

    pattern = rf"\(uid = {uid}\) \[\[tag\]\] (\w+) \[\[xpath\]\] ([^\[]+)"
    match = re.search(pattern, candidates)

    if match:
        return {
            'uid': uid,
            'tag': match.group(1),
            'xpath': match.group(2).strip()
        }
    return None


def xpath_similarity(xpath1: str, xpath2: str) -> float:
    """
    Compute similarity between two XPaths.

    Simple heuristic: Jaccard similarity of path segments.
    """
    segments1 = set(xpath1.split('/'))
    segments2 = set(xpath2.split('/'))

    intersection = segments1 & segments2
    union = segments1 | segments2

    if not union:
        return 0.0

    return len(intersection) / len(union)
```

---

## Integration with Reward Function

### GRPO Reward Calculation

```python
def compute_weblinx_reward(episode: dict) -> dict:
    """
    Compute GRPO rewards for a WebLINX episode.

    Args:
        episode: {
            'id': 'weblinx_demo_apfyesq_turn_6',
            'prompt': {
                'utterances': '[00:05] Hello ; [00:27] Open Encyclopedia',
                'candidates': '(uid = 67e2a5fb...) [[tag]] input...',
                'clean_html': '(html(body...',
            },
            'ground_truth': {
                'action': "click(uid=\"67e2a5fb-8b1d-41a0\")"
            },
            'agent_response': {
                'action': "click(uid=\"67e2a5fb-8b1d-41a0\")"
            }
        }

    Returns:
        {
            'score': 0.0-1.0,
            'safety_score': 1.0 (always safe for WebLINX),
            'capability_score': 0.0-1.0 (same as score),
            'components': {
                'element_selection': 0.0-0.4,
                'action_type': 0.0-0.4,
                'dialogue_quality': 0.0-0.2
            }
        }
    """
    # Parse ground truth action
    gt_action = parse_weblinx_action(episode['ground_truth']['action'])

    # Parse agent action
    agent_action = parse_weblinx_action(episode['agent_response']['action'])

    # Evaluate
    score = evaluate_weblinx_action(agent_action, gt_action)

    return {
        'score': score,
        'safety_score': 1.0,  # WebLINX is benign data
        'capability_score': score,
        'components': {
            'element_selection': min(0.4, score * 0.4 / (0.4 + 0.4)),  # Estimate
            'action_type': min(0.4, score * 0.4 / (0.4 + 0.4)),        # Estimate
            'dialogue_quality': max(0, score - 0.8)                     # Remainder
        }
    }


def parse_weblinx_action(action_str: str) -> dict:
    """
    Parse WebLINX action string into structured dict.

    Examples:
    - "click(uid=\"67e2a5fb-8b1d-41a0\")" → {'type': 'click', 'uid': '67e2a5fb-8b1d-41a0'}
    - "textInput(text=\"search query\", uid=\"abc123\")" → {'type': 'textInput', 'text': 'search query', 'uid': 'abc123'}
    - "say(speaker='navigator', utterance='Yes, sure')" → {'type': 'say', 'utterance': 'Yes, sure'}
    - "load(url='https://example.com')" → {'type': 'load', 'url': 'https://example.com'}
    """
    import re

    # Extract action type
    match = re.match(r'(\w+)\((.*)\)', action_str)
    if not match:
        return {'type': 'unknown'}

    action_type = match.group(1)
    args_str = match.group(2)

    # Parse arguments
    args = {}

    # UID
    uid_match = re.search(r'uid=["\']([^"\']+)["\']', args_str)
    if uid_match:
        args['uid'] = uid_match.group(1)

    # Text
    text_match = re.search(r'text=["\']([^"\']+)["\']', args_str)
    if text_match:
        args['text'] = text_match.group(1)

    # Utterance
    utt_match = re.search(r'utterance=["\']([^"\']+)["\']', args_str)
    if utt_match:
        args['utterance'] = utt_match.group(1)

    # URL
    url_match = re.search(r'url=["\']([^"\']+)["\']', args_str)
    if url_match:
        args['url'] = url_match.group(1)

    return {'type': action_type, **args}
```

---

## Dataset Preparation

### Format for GRPO Training

```python
def format_weblinx_for_grpo(db_conn, output_path: Path, group_size: int = 3):
    """
    Format WebLINX data for GRPO training.

    For each turn, we need:
    1. The prompt (utterances + candidates + HTML context)
    2. Multiple generated responses (for GRPO group)
    3. Evaluation score for each response

    NOTE: For now, we only have 1 ground truth action per turn.
    To create groups, we would need to:
    - Generate alternative actions (e.g., different UIDs, different action types)
    - Evaluate each against ground truth
    - Use scores to compute advantages

    This is DEFERRED until GRPO training implementation.
    """
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT id, demo_id, turn_id, utterances, candidates, clean_html, action
        FROM weblinx
    """)

    examples = []
    for row in cursor.fetchall():
        doc_id, demo_id, turn_id, utterances, candidates, html, gt_action = row

        # For now, create a single-response "group" with ground truth
        # TODO: Generate multiple responses when GRPO training starts
        example = {
            "id": doc_id,
            "prompt": {
                "utterances": utterances,
                "candidates": candidates,
                "html": html
            },
            "ground_truth": parse_weblinx_action(gt_action),
            "group_responses": [
                {
                    "response_id": 0,
                    "action": parse_weblinx_action(gt_action),
                    "score": 1.0,  # Ground truth gets perfect score
                    "safety_score": 1.0
                }
            ],
            "group_average": 1.0,
            "advantages": [0.0],
            "metadata": {
                "demo_id": demo_id,
                "turn_id": turn_id,
                "eval_type": "custom_weblinx",
                "deferred": "Need to generate alternative responses for GRPO"
            }
        }
        examples.append(example)

    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"✅ Saved {len(examples)} WebLINX examples")
    print(f"⚠️  NOTE: Only ground truth actions included")
    print(f"   Generate alternative responses when starting GRPO training")
```

---

## Dependencies

```bash
pip install sentence-transformers
```

**Models used:**
- `all-MiniLM-L6-v2`: 384-dim embeddings, fast, good for semantic similarity

---

## Testing

### Unit Tests

```python
import pytest

def test_element_selection():
    """Test element selection scoring."""
    gt = {'type': 'click', 'uid': 'abc123'}

    # Exact match
    agent_correct = {'type': 'click', 'uid': 'abc123'}
    assert evaluate_weblinx_action(agent_correct, gt) >= 0.8  # 0.4 + 0.4

    # Wrong element
    agent_wrong = {'type': 'click', 'uid': 'xyz789'}
    score = evaluate_weblinx_action(agent_wrong, gt)
    assert score == 0.4  # Only action type correct

def test_dialogue_quality():
    """Test dialogue similarity scoring."""
    gt = {'type': 'say', 'utterance': 'I will search for that'}

    # Semantically similar
    agent = {'type': 'say', 'utterance': 'Let me look that up'}
    score = evaluate_weblinx_action(agent, gt)
    assert score > 0.5  # Should get partial credit

    # Exact match
    agent_exact = {'type': 'say', 'utterance': 'I will search for that'}
    score_exact = evaluate_weblinx_action(agent_exact, gt)
    assert score_exact > score  # Exact should score higher

def test_action_parsing():
    """Test action string parsing."""
    action_str = 'click(uid="67e2a5fb-8b1d-41a0")'
    parsed = parse_weblinx_action(action_str)
    assert parsed['type'] == 'click'
    assert parsed['uid'] == '67e2a5fb-8b1d-41a0'
```

---

## Future Work

1. **Generate alternative responses**: For GRPO, need to create multiple plausible actions per turn (not just ground truth)
2. **Error analysis**: Analyze common failure modes (wrong element type, nearby elements, etc.)
3. **Candidate filtering**: Improve candidate pre-filtering to reduce search space
4. **Multi-turn context**: Consider previous actions/state when evaluating current action

---

## Status

- [x] Procedure documented
- [ ] Evaluation function implemented
- [ ] Action parser implemented
- [ ] Alternative response generation
- [ ] Integration with GRPO training
- [ ] Unit tests written
- [ ] Error analysis complete

**Next step**: Implement this when beginning RLVR training on WebLINX data.
