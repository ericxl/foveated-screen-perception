# Foveated Screen Perception for Computer-Use Agents

## The problem

Current computer-use agents send a full screenshot (~1,300 tokens on Claude, ~765-1,105 on GPT-4o) to the LLM at every step. A 50-step task burns ~65K tokens on screenshots alone -- often 80%+ of context. The actionable information per step is tiny: "button clicked, dialog appeared."

This is not just wasteful -- it actively hurts performance. The OSWorld-Human study (ICML 2025) found that **planning and reflection calls consume 75-94% of total task latency**, with each successive step taking up to 3x longer due to growing context. Perhaps most critically: **screenshot-only history does not improve performance -- only text-based history helps.** Agents take 1.4-2.7x more steps than human-optimal trajectories, and the dominant failure mode across OSWorld, ScreenSpot-Pro, and WindowsAgentArena is **GUI grounding errors** -- misclicking or selecting wrong elements.

Human vision solves the bandwidth problem with five mechanisms: foveation (high-res center, blurry periphery), saccades (targeted jumps), top-down attention (task-biased filtering), predictive coding (only process what's unexpected), and gist extraction (scene category in 36ms). The human retina sends ~10^9 bits/sec but only ~10^7 bits/sec reach cortex -- a 100x compression before conscious processing ever begins.

## What the field has tried (prior work)

### The observation landscape: how current systems look at screens

**Almost every production system sends full screenshots.** OpenAI Operator/CUA, Google Mariner, UI-TARS/UI-TARS-2 (ByteDance), and SeeAct-V all follow the same loop: capture full screenshot -> send to VLM -> receive action -> execute -> repeat. The model receives the full viewport each turn with no crop, zoom, or mode selection.

Two exceptions stand out:

- **Claude Computer Use** (`computer_20251124`): Adds a `zoom` action that returns a cropped, upscaled region. The only production API where the model itself can choose to look at a sub-region. Claude's OSWorld score reached ~72.5% (late 2025), roughly 2x the previous best.
- **GUI-Eyes** (Jan 2026): The first system where the agent is RL-trained to decide whether and how to invoke perception tools (`crop` and `zoom`). Reward: R = 0.6*R_acc + 0.1*R_format + 0.3*R_tool, where R_tool combines spatial proximity and region IoU. A 3B model with GUI-Eyes reaches 44.8% on ScreenSpot-Pro -- vs 30.2% without.

### Zoom/crop is a free lunch for grounding

The evidence is now overwhelming that letting agents zoom into regions dramatically improves grounding, with no retraining required:

| System | Approach | ScreenSpot-Pro Improvement |
|--------|----------|---------------------------|
| **RegionFocus** (ICCV 2025) | Iterative zoom with multiple aspect-ratio crops; image-as-map stamps pink stars at prior attempts | UI-TARS-72B: 38.1% -> 50.2% (+31.7%); Qwen2.5-VL-72B: 47.8% -> 61.6% (+28.9%) |
| **ScreenSeeker** (2025) | GPT-4o as planner -> cascaded recursive cropping -> OS-Atlas-7B grounds in patch | OS-Atlas-7B: 18.9% -> 48.1% (+254% relative) |
| **ZoomClick** (Dec 2025, Princeton) | Training-free 3-stage zoom. Pre-zoom on 2x2 grid, depth=3, shrink=0.5x per iter, min crop 768px | UI-Venus-72B: 61.4% -> 73.1% (SOTA at publication) |
| **AdaZoom-GUI** (Mar 2026) | GRPO-trained conditional zoom -- only zooms on small predicted elements + instruction refinement via Qwen3.5 | 61.6% -> 76.8% (current SOTA) |
| **GUI-ARP** (Sep 2025) | Attention-map-guided cropping with adaptive stage control. Agent decides 1-stage vs 2-stage | 60.8% with 7B model (beats UI-TARS-72B at 38.1%) |
| **LASER** (Sep 2025) | Self-evolving preference optimization for crop policy via Monte Carlo rollouts | Qwen2.5-VL-7B: 26.8% -> 47.5% (+20.7pp) |
| **SpiritSight** (CVPR 2025) | Universal Block Parsing: block-local coordinates in 448x448 patches, eliminating coordinate ambiguity | 80.7% AMS on AMEX, 87.6% on AndroidControl |

The small-element problem is the core driver: ScreenSpot-Pro targets occupy only **0.07%** of image area (vs 2.01% in older benchmarks). Zooming eliminates irrelevant visual context around tiny targets -- the single highest-leverage intervention for grounding.

### Accessibility-tree vs vision vs hybrid

The structured-data vs vision debate has resolved into a clear hierarchy:

**Accessibility-first (DirectShell, Feb 2026):** 50-200 tokens/step, 10-25x cheaper than screenshots. 85% task success when a11y trees are complete. But Screen2AX (MacPaw, Jul 2025) audited 302 macOS apps and found **only 33% provide complete native accessibility**. DirectShell breaks on custom UIs, canvas, games, Google Search (poor a11y semantics), Booking.com (27.3% success), Google Flights (35.7%).

**Vision-first (UI-TARS, Operator):** Universal compatibility. 1,200-5,000 tokens/step. Coordinate precision issues on small elements.

**Hybrid (UFO2/Microsoft, Agent S2/Simular):** Best real-world results. UFO2 queries Windows UI Automation + OmniParser-v2 (YOLOv8 + Florence-2) to fill gaps. Agent S2 uses a Mixture of Grounding Experts (visual + OCR + structural). **OmniParser-v2** converts screenshots to structured elements at 0.6s/frame on A100, achieving 73% accuracy improvement over baseline.

**Browser tasks:** Accessibility trees are dominant and reliable (DOM/a11y well-structured by definition). Browser-Use achieves 89.1% on WebVoyager with a11y-first observation.

**The consensus:** Use structured data when available and complete; fall back to vision when it isn't; fuse both when you can. The agent should not be locked into one modality.

### Context compression: the hidden performance lever

The growing context of historical observations is what causes exponential latency degradation. Recent work attacks this from multiple angles:

- **Fara-7B** (Microsoft, Nov 2025): Keeps only last 3 screenshots in history, retains all text. Completes tasks in ~16 steps vs ~41 for UI-TARS-1.5-7B. The key insight: text-based history is load-bearing; screenshot history is not.
- **JetBrains "Complexity Trap"** (NeurIPS DL4Code 2025): Simple **observation masking** (replacing old tool outputs with placeholders like "[Previous output: 2,847 tokens, masked]") halves cost while matching or slightly exceeding LLM summarization. In SE agent turns, observation dominates token count, and a placeholder is often better than a bad summary.
- **AgentOCR** (Jan 2026): Renders text history as images, exploiting the fact that VLMs encode visual tokens more densely. Retains 99.5% performance while cutting tokens by >50%. Agent selects compression rate via RL.
- **ACON** (Oct 2025): Optimizes natural-language compression guidelines via gradient-free optimization. 26-54% peak token reduction.
- **Grouped-action trajectories** (OSWorld-Human): Multiple consecutive actions can execute from a single observation because the UI doesn't change between them. Top agents waste 1.4-2.7x more observation cycles than theoretically necessary.

**The principle:** Vision for the present, text for the past. Let the agent decide how much of each.

### Bio-inspired models: what exists

- **Foveated vision transformers** (FovealNet, log-polar ViTs): 30-50% FLOP reduction on ImageNet. None applied to GUI/screen understanding yet.
- **Predictive coding networks** (PC-Transformers, Salvatori et al. ICLR 2024): Process only prediction errors. Directly maps to the "diff" concept but no one has used it for agent observation.
- **Active vision** (RAM descendants, Active Vision Transformers): Sequential fixation policies via RL. The GUI-Eyes system is the closest to this in practice.
- **Task-conditioned visual encoding** (Qwen2.5-VL, InternVL 2.5): Text prompt modulates visual processing from early layers. Theoretically ideal for GUI agents but underexploited -- could encode "click the Submit button" differently than "read the error message."
- **Visual working memory in AI:** Nearly nonexistent for GUI agents. Standard systems pile screenshots into the token window with no compression. Your `SimpleMemory` (compress old steps to one-liners, keep recent in full) has no direct precedent in the GUI agent literature.

**The gap:** Nobody has published a system where the agent explicitly controls its observation modality via tool calls. Claude's `zoom` is the closest production feature. GUI-Eyes trains the decision via RL. Our proposed `observe` tool with mode selection would be novel. The further insight -- that most modes should return **text only**, with image crops as the rare exception -- has no precedent. Every existing system treats the screenshot as the primary observation; we treat text as primary and pixels as fallback.

---

## The design: LLM chooses how to look

Instead of force-feeding the LLM a screenshot every step, we give it an `observe` tool with a `mode` parameter. **The LLM is the CEO -- it decides what kind of observation it needs.**

### Why these modes? Mapping biology to engineering

The modes are designed to mirror the **Guess-Scan-Confirm** cycle observed in eye-tracking studies of GUI interaction (84 participants, 10,282 trials across 900 GUIs). Humans follow a stereotyped sequence: orient to the scene (ambient mode, ~700ms) -> search for the target (focal mode, guided search) -> verify the action succeeded (predictive coding / confirmation fixation). Each mode maps to a phase:

| Phase | Bio Mechanism | Agent Implementation | What It Returns | Cost |
|-------|--------------|---------------------|-----------------|------|
| **Orient** | Gist extraction (36-100ms) + ambient mode (large saccades, short fixations, dorsal "where" stream) | `glance` mode | **Text only.** App name, screen type, layout structure, overall state. No image -- humans store a semantic gist, not a low-res snapshot. | ~30-50 tokens |
| **Read** | Foveal text reading -- serial word recognition at fixation point. The dominant mode of human screen interaction. | `read` mode | **Text only.** Extracted text content at/around (x, y) coordinates via OCR or a11y. Most screen use is text comprehension; visual encoding is just a means to this end. | ~20-100 tokens |
| **Inspect** | Foveal fixation on non-text elements -- icons, spatial relationships, color cues, charts. Ventral "what" stream. | `look` mode | **Image crop** around (x, y). The only mode that returns pixels. Used when text extraction isn't enough. | ~50-150 tokens |
| **Confirm** | Predictive coding (Rao & Ballard 1999). Brain generates prediction -> only prediction errors propagate. Object-level, not pixel-level | `check` mode | **Text only.** Semantic diff anchored to action point. Agent states expectation; system reports whether met or violated. | ~10-50 tokens |

### Why NOT `elements` as a primary mode

The accessibility tree has **no biological analog**. The brain doesn't have a "list all objects" mode -- it searches for specific targets (Wolfe's Guided Search) or identifies what's at a specific fixation point (focal recognition). Making `elements` a top-level mode encourages the anti-pattern of dumping 200-300 tokens of structured data every step regardless of need.

Instead, accessibility data is an **implementation detail** of how `read` and `glance` work internally:
- `glance` uses a11y tree (when available) to generate its layout description. Falls back to screenshot + OCR when a11y is absent.
- `read` uses a11y text content at the target coordinates when available. Falls back to OCR on a cropped region.
- `look` uses a11y element data to annotate the crop (if an element at the target coordinates has a11y metadata, include it). Falls back to pure vision.
- A `find` mode (V2) would use a11y for lookup by name, falling back to visual search.

The agent never needs to think about whether it's getting accessibility data or vision data. It asks "what's here?" and the system gives it the best available answer.

### Why NO `scan` mode

The original design included a `scan` mode (full screenshot at reduced resolution, ~300-400 tokens) as a "recovery" option. This was dropped because:

1. **No biological analog.** The brain never processes the entire visual field at uniform high resolution. "See everything at once" is exactly the anti-pattern the naive screenshot approach already exhibits.
2. **`glance` covers recovery.** An agent that is "lost" needs to re-orient -- that's what `glance` does. If more detail is needed, targeted `read` or `look` calls at specific regions are more efficient than dumping the whole screen.
3. **It's a crutch that undermines the design.** If `scan` is available, agents will default to it instead of learning to use cheaper modes. Removing it forces the agent to develop better perception habits.

### Why `read` is a first-class mode

Most human screen interaction is **text comprehension**: reading labels, menus, error messages, form fields, status text. When you read text on a screen, the visual processing is just a means to an end -- the information being extracted *is* text. Having a mode that returns text directly (via OCR or a11y) skips the visual encoding step entirely.

This is the mode that was most conspicuously missing from the original design. Three of four modes now return **text only**, with `look` (image crop) as the exception used only when spatial/visual information can't be captured as text (unlabeled icons, color indicators, charts, spatial layout relationships).

### Why `check` is NOT a pixel diff

Biological predictive coding is fundamentally different from SSIM or perceptual hashing:

- The brain generates **hierarchical predictions**: higher cortex sends "I expect a dialog box here" downward; lower areas compare against actual input and send **prediction errors** upward.
- Errors are **semantic, not pixel-level**: surprise in one feature of an object spreads to make the entire object unexpected. Even V1 responds to high-level prediction violations.
- When predictions match, neural activity is **suppressed** (repetition suppression). The default is "nothing to report."

`check` mirrors this: the agent states what it expected ("a save dialog should appear"), and the system reports whether reality matched or diverged. The implementation can use pixel-level diff for initial change detection, but the output is always semantic: "Dialog appeared as expected" or "No dialog appeared -- screen unchanged. The Save button is still visible at [e12]."

This is anchored to the action point because eye-tracking shows humans fixate near where they just clicked for verification. Changes far from the action point are routinely missed -- **40% of error messages go unnoticed** when they appear away from the user's focus (NN/g change blindness study).

### The `observe` tool schema

```python
OBSERVE_TOOL = {
    "name": "observe",
    "description": (
        "Observe the current screen state. Choose the cheapest mode "
        "that gives you what you need. Three modes return text only; "
        "one returns an image.\n\n"
        "Modes (cheapest to most expensive):\n"
        "- 'check': Did my last action work? (~10-50 tokens, text only). "
        "  Provide your expectation. Best after routine actions.\n"
        "- 'read': What does the text say here? (~20-100 tokens, text only). "
        "  Extracts text content at/around coordinates. Best for labels, "
        "  menus, error messages, form fields -- most screen interaction.\n"
        "- 'glance': What am I looking at? (~30-50 tokens, text only). "
        "  App type, layout, state. Best when arriving at a new screen.\n"
        "- 'look': What does this look like? (~50-150 tokens, image). "
        "  High-res crop. Only mode that returns pixels. Best for icons, "
        "  colors, charts, spatial relationships that text can't capture."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["check", "read", "glance", "look"],
            },
            "center_x": {
                "type": "integer",
                "description": "For 'read'/'look': X center of region"
            },
            "center_y": {
                "type": "integer",
                "description": "For 'read'/'look': Y center of region"
            },
            "radius": {
                "type": "integer",
                "default": 150,
                "description": "For 'read'/'look': half-width of region in pixels"
            },
            "expectation": {
                "type": "string",
                "description": "For 'check': what you expected to happen"
            }
        },
        "required": ["mode"]
    }
}
```

### How the modes flow (the agent loop)

```
New screen or navigation
  -> glance (orient: "what am I looking at?" -- text only)
  -> read (extract text near target: "what does this say?" -- text only)
  -> act (click, type, scroll, keyboard shortcut)
  -> check (verify: "I expected the form to submit" -- text only)
    -> if check says "as expected": proceed to next action
    -> if check says "unexpected": read or look at what changed
    -> if check says "screen unchanged": action failed, try alternative
  -> repeat
```

`look` (the only visual mode) is the exception, not the default -- used when check/read surface something that needs spatial or visual inspection:

```
  -> check says "unexpected"
  -> read near action point (get text of what appeared)
  -> if text is enough to understand: proceed
  -> if need visual detail (icon, color, layout): look at the region
```

This mirrors the biological Orient-Read-Confirm cycle:
- `glance` = Orient phase (gist extraction: scene category in 36ms, dorsal "where" stream)
- `read` = Comprehend phase (foveal text reading: serial word recognition at fixation)
- `check` = Confirm phase (predictive coding: did reality match my expectation?)
- `look` = Inspect phase (foveal fixation on non-text elements: ventral "what" stream, used only when needed)

---

## V1 implementation

### What to build

One skill named `computer_use`, two perception backends selected by config:

```
fsp/
  computer_use/
    __init__.py       # ComputerUse(perception="standard"|"fsp") — single entry point
    actions.py        # click, type_text, scroll, keyboard_shortcut (shared by both)
    capture.py        # screen capture, a11y queries, OCR (shared by both)
    memory.py         # Sliding window: keep last N steps as text, drop old images

    perception/
      standard.py     # Full screenshot every step. The control condition.
      fsp.py          # observe tool: check/read/glance/look. The treatment.
      observe.py      # Mode dispatch for FSP backend
      diff.py         # Change detection for 'check' mode (phash + SSIM + semantic)
      glance.py       # a11y layout description (with OCR fallback). Text only.
      read.py         # Text extraction at coordinates via a11y or OCR crop

  eval/
    runner.py         # Takes (model, perception, benchmark) → results
    screenspot.py     # ScreenSpot-Pro loader
    osworld.py        # OSWorld loader
```

The model always sees a skill called `computer_use`. The perception backend is a config flag:

```python
# Evaluation runs
for model in ["gpt-5.4", "opus-4.6"]:
    for perception in ["standard", "fsp"]:
        agent = ComputerUse(model=model, perception=perception)
        results = runner.run(agent, benchmark="screenspot-pro")
```

**Standard backend:** Model gets a single `computer_use` tool. Every action returns a full screenshot. Mirrors current Claude/OpenAI production behavior.

**FSP backend:** Model gets `computer_use` with an `observe` sub-tool. Actions and observations are decoupled. The model decides when and how to look.

### `observe.py` core

```python
from PIL import Image
import imagehash

class Observer:
    def __init__(self):
        self.prev_screenshot = None
        self.prev_hash = None
        self.last_action_point = None  # (x, y) of last click/interaction

    def observe(self, mode: str, **kwargs) -> dict:
        """LLM chooses observation mode. Three modes return text only; one returns an image."""
        if mode == "check":
            return self._check(kwargs.get("expectation", ""))
        elif mode == "glance":
            return self._glance()
        elif mode == "read":
            return self._read(kwargs["center_x"], kwargs["center_y"],
                              kwargs.get("radius", 150))
        elif mode == "look":
            return self._look(kwargs["center_x"], kwargs["center_y"],
                              kwargs.get("radius", 150))

    def _glance(self) -> dict:
        """Scene gist: text-only layout description. No image returned."""
        img = capture_screen()
        self._update_state(img)
        # Try a11y for layout description; fall back to OCR
        layout = try_accessibility_layout() or ocr_layout_description(img)
        return {
            "type": "glance",
            "text": layout,  # "Chrome - Gmail Inbox. Sidebar left, email list center, reading pane right."
            "tokens": ~30-50
        }

    def _read(self, cx: int, cy: int, radius: int) -> dict:
        """Foveal text reading: extract text at a point. Text only, no image."""
        img = capture_screen()
        self._update_state(img)
        # Try a11y text content at coordinates; fall back to OCR on crop
        text = try_accessibility_text_at(cx, cy, radius)
        if text is None:
            crop = img.crop((
                max(0, cx - radius), max(0, cy - radius),
                min(img.width, cx + radius), min(img.height, cy + radius)
            ))
            text = ocr_extract_text(crop)
        return {
            "type": "read",
            "text": text,
            "tokens": len(text.split()) // 2  # rough estimate
        }

    def _look(self, cx: int, cy: int, radius: int) -> dict:
        """Foveal fixation: high-res crop at a point. The only mode that returns an image."""
        img = capture_screen()
        self._update_state(img)
        crop = img.crop((
            max(0, cx - radius), max(0, cy - radius),
            min(img.width, cx + radius), min(img.height, cy + radius)
        ))
        annotations = try_accessibility_in_region(cx, cy, radius)
        return {
            "type": "look",
            "image": crop,
            "annotations": annotations,
            "tokens": (crop.width * crop.height) // 750
        }

    def _check(self, expectation: str) -> dict:
        """Predictive coding: expectation vs reality, anchored to action point. Text only."""
        img = capture_screen()
        curr_hash = str(imagehash.phash(img))

        if self.prev_hash is None:
            self._update_state(img)
            return {"type": "check", "text": "First observation.", "tokens": 5}

        distance = imagehash.hex_to_hash(curr_hash) - imagehash.hex_to_hash(self.prev_hash)
        self._update_state(img)

        if distance < 3:
            return {
                "type": "check",
                "text": f"Screen unchanged. Expected: {expectation}. "
                        "Action may have failed.",
                "tokens": ~15
            }

        # Describe what changed semantically, focused near action point
        changes = describe_changes_near(
            self.prev_screenshot, img, self.last_action_point
        )
        met = does_change_match_expectation(changes, expectation)
        return {
            "type": "check",
            "text": f"Expected: {expectation}. "
                    f"{'Confirmed.' if met else 'NOT as expected.'} "
                    f"Changes: {changes}",
            "tokens": ~20-50
        }

    def _update_state(self, img):
        self.prev_screenshot = img
        self.prev_hash = str(imagehash.phash(img))
```

### Memory: vision for the present, text for the past

Following Fara-7B's finding that text history outperforms screenshot history, and JetBrains' finding that simple masking beats summarization:

```python
class SimpleMemory:
    def __init__(self, window=5):
        self.steps = []
        self.window = window

    def add(self, step_num: int, action: str, observation_summary: str):
        self.steps.append({"step": step_num, "action": action, "obs": observation_summary})

    def get_context(self) -> str:
        """Older steps: one-line text. Recent steps: full detail in conversation."""
        lines = []
        for s in self.steps[:-self.window]:
            lines.append(f"Step {s['step']}: {s['action']} -> {s['obs']}")
        return "\n".join(lines)
```

### System prompt

```
You are a desktop automation agent with efficient perception.

You control how you observe the screen via the `observe` tool.
Three modes return text only; one returns an image. Pick the cheapest
mode that gives you what you need:

ORIENT: When you arrive at a new screen, use mode="glance".
  Text description of app type, layout, and state (~30 tokens).

READ: When you need to know what text says, use mode="read" with
  coordinates. Returns extracted text content (~20-100 tokens).
  This is the most common mode -- most screen interaction is reading.

VERIFY: After each action, use mode="check" with your expectation.
  Text confirmation of whether reality matched (~20 tokens).

INSPECT: Only when you need visual detail that text can't capture
  (icons, colors, charts, spatial layout), use mode="look" with
  coordinates. This is the only mode that returns an image (~100 tokens).

Prefer text modes (check, read, glance) over look.
Prefer "check" after actions -- it's the cheapest verification.
If "check" says the screen is unchanged, your action likely failed --
try a different approach rather than repeating.
```

---

## Expected token savings

| Step pattern | Naive (screenshot/step) | FSP (LLM chooses) |
|---|---|---|
| Arrive at new screen, orient | 1,300 | ~40 (glance, text only) |
| Read a label/menu/error message | 1,300 | ~40 (read, text only) |
| Verify action succeeded | 1,300 | ~20 (check, text only) |
| Inspect a visual element (icon, chart) | 1,300 | ~100 (look, image crop) |
| **Typical 50-step task** | **~65,000** | **~2,000-3,500** |

Estimated **19-33x token reduction.** The improvement over the original FSP design (~3-5K) comes from replacing most `look` calls with text-only `read` calls and dropping the `scan` fallback entirely. Compatible with existing context compression (Fara-7B's "keep last 3", JetBrains observation masking) for further gains.

**Token profile of a typical task:** Most steps use `check` (~15 tokens) or `read` (~40 tokens). Occasional `glance` on screen transitions (~40 tokens). Rare `look` when visual detail is needed (~100 tokens). Three of four modes are text-only.

---

## Evaluation

### Experimental design: 2x2 matrix

The claim is that FSP improves perception efficiency **regardless of the underlying model**. To make this conclusive, we test two models x two perception approaches:

| | Standard CU (full screenshot every step) | FSP (`observe` tool) |
|---|---|---|
| **GPT-5.4** (OpenAI) | Baseline A | Treatment A |
| **Opus 4.6** (Anthropic) | Baseline B | Treatment B |

**The result is conclusive if FSP wins on both rows.** This controls for the objection "maybe your tool just suits one model's training." If both GPT-5.4 and Opus 4.6 benefit, the perception interface itself is better, not model-specific tuning.

### Benchmarks

**Phase 1: ScreenSpot-Pro (grounding accuracy, fast iteration)**
- Static dataset, no VM needed. Quick turnaround.
- Measures: can the agent click the right element?
- Key difficulty: targets occupy only 0.07% of image area -- where zoom/crop approaches shine.
- Run all 4 conditions (2 models x 2 approaches).

**Phase 2: OSWorld (end-to-end task completion)**
- Full desktop tasks in VMs. Slower, more expensive.
- Measures: does the agent complete real tasks?
- The benchmark most cited in the CLAUDE.md research review (latency degradation, observation waste, grounding failures).
- Run all 4 conditions.

### Metrics per condition

| Metric | What it tells us |
|--------|-----------------|
| Success rate | FSP must not hurt task completion |
| Total input tokens per task | The core claim: 19-33x reduction |
| Observation tokens only | Isolates perception savings from reasoning tokens |
| Steps to completion | Fewer wasted observation cycles (OSWorld-Human found agents use 1.4-2.7x more than necessary) |
| Mode distribution (FSP only) | What does the agent actually choose? How often does it fall back to `look`? |
| Latency per step | Less context = faster inference |
| Grounding accuracy (ScreenSpot-Pro) | Click precision on small elements |

### Implementation

Both conditions use the same `computer_use` skill. The only variable is the perception backend:

- **`perception="standard"`:** Full screenshot returned after every action. Model sees one `computer_use` tool. Mirrors current Claude/OpenAI production behavior.
- **`perception="fsp"`:** Screenshot captured internally but NOT sent directly. Model sees `computer_use` with `observe` sub-tool (4 modes). Actions and observations are decoupled — the model decides when and how to look.

Same skill name, same action space, same system prompt (minus perception-specific instructions). The model doesn't know which condition it's in.

---

## What's NOT in V1 (but validated by research for V2+)

| Feature | Research basis | V2 priority |
|---------|---------------|-------------|
| `find` mode (guided search by target description) | Wolfe's Guided Search; ScreenSeeker's cascaded search (+254%) | High -- replaces blind `elements` dump |
| RL-trained mode selection | GUI-Eyes: 0.6*R_acc + 0.3*R_tool reward; LASER: Monte Carlo preference optimization | High -- agent learns when to zoom vs skip |
| Grouped-action execution | OSWorld-Human: agents waste 1.4-2.7x observation cycles | Medium -- multiple actions from one observation |
| Task-conditioned visual encoding | Qwen2.5-VL cross-attention; InternVL prompt-modulated vision | Medium -- encode screenshot differently per task |
| Hierarchical memory consolidation | AgentOCR: RL-selected compression; ACON: NL-optimized guidelines | Medium -- adaptive compression of old steps |
| OmniParser fallback for broken a11y | Screen2AX: only 33% of macOS apps have full a11y; OmniParser v2: 0.6s/frame | Medium -- vision-generated element lists |
| Predictive screen state model | PC-Transformers (ICLR 2024); bio predictive coding | Low -- richer `check` with hierarchical prediction |

---

## The core bet

Every production computer-use system today sends a full screenshot at every step. The research unanimously shows this is wasteful: Fara-7B proves text history >> screenshot history, JetBrains proves old observations can be masked with zero cost, OSWorld-Human proves agents take 1.4-2.7x more observation cycles than necessary, and the entire zoom/crop literature proves targeted inspection outperforms full-screen processing.

The bet is: **an LLM that can choose how to look will look more efficiently than one forced to see everything -- and most of the time, it doesn't need to "look" at all, just read.** The biological evidence (ambient-to-focal transition at 700ms, 40% change blindness for off-focus events, 36ms gist extraction, predictive coding suppression of expected input) explains why this should work. The agent systems evidence (GUI-Eyes, RegionFocus, ZoomClick, AdaZoom-GUI) proves it works in practice for grounding. What's missing is a unified tool that packages all observation modes into a single, simple interface the LLM can drive -- with text as the default output and pixels as the exception.

That's what `observe` with `check/read/glance/look` is. Three text-only modes, one image mode.
