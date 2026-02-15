# Foveated Screen Perception for Computer-Use Agents

## The problem

Current computer-use agents send a full screenshot (~1,300 tokens on Claude, ~765-1,105 on GPT-4o) to the LLM at every step. A 50-step task burns ~65K tokens on screenshots alone -- often 80%+ of context. The actionable information per step is tiny: "button clicked, dialog appeared."

This is not just wasteful -- it actively hurts performance. The OSWorld-Human study (ICML 2025) found that **planning and reflection calls consume 75-94% of total task latency**, with each successive step taking up to 3x longer due to growing context. Perhaps most critically: **screenshot-only history does not improve performance -- only text-based history helps.** Agents take 1.4-2.7x more steps than human-optimal trajectories, and the dominant failure mode across OSWorld, ScreenSpot-Pro, and WindowsAgentArena is **GUI grounding errors** -- misclicking or selecting wrong elements.

Human vision solves the bandwidth problem with a particular architectural trick: **the retina does not send one signal to cortex, it sends at least three**. Rods (~120M cells) drive a low-resolution grayscale wide-field stream (magnocellular, dorsal, "where"); cones (~6M cells) drive a high-resolution color stream concentrated at the fovea (parvocellular, ventral, "what"); and text reading is handled by a specialized cortical area (the Visual Word Form Area) that extracts symbolic content from the ventral stream. This decomposition achieves a ~100x bandwidth reduction (~10^9 bits/sec retinal to ~10^7 bits/sec cortical) before any conscious processing begins.

The three streams are not redundant, they are complementary. The magno/rod stream is cheap precisely because it drops color and resolution. The parvo/cone stream pays for color and resolution only where it matters (the fovea). The VWFA bypasses pixels entirely for text. A computer-use agent that sends one full-color full-resolution frame every step is discarding the compression strategy evolution spent half a billion years tuning.

## What the field has tried (prior work)

### The observation landscape: how current systems look at screens

**Almost every production system sends full screenshots.** OpenAI Operator/CUA, Google Mariner, UI-TARS/UI-TARS-2 (ByteDance), and SeeAct-V all follow the same loop: capture full screenshot -> send to VLM -> receive action -> execute -> repeat. The model receives the full viewport each turn with no crop, zoom, or mode selection.

Two exceptions stand out:

- **Claude Computer Use** (`computer_20251124`): Adds a `zoom` action that returns a cropped, upscaled region. The only production API where the model itself can choose to look at a sub-region. Claude's OSWorld score reached ~72.5% (late 2025), roughly 2x the previous best.
- **GUI-Eyes** (Jan 2026): The first system where the agent is RL-trained to decide whether and how to invoke perception tools (`crop` and `zoom`). Reward: R = 0.6*R_acc + 0.1*R_format + 0.3*R_tool, where R_tool combines spatial proximity and region IoU. A 3B model with GUI-Eyes reaches 44.8% on ScreenSpot-Pro -- vs 30.2% without.

Every one of these systems stays within a single image modality. None exposes a color-vs-grayscale choice, a peripheral-vs-foveal choice, or a pixels-vs-text choice as a tool parameter.

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

The small-element problem is the core driver: ScreenSpot-Pro targets occupy only **0.07%** of image area (vs 2.01% in older benchmarks). Zooming eliminates irrelevant visual context around tiny targets -- the single highest-leverage intervention for grounding. But these systems all operate at full color fidelity. None has asked whether color is even needed for the "where is my target?" step that precedes the crop.

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
- **Two-stream architectures**: Simonyan & Zisserman's two-stream networks (spatial + temporal) for action recognition are structurally analogous to magno/parvo, but the split is along a different axis (RGB vs optical flow) and the agent never sees the streams -- they are fused internally.
- **Visual working memory in AI:** Nearly nonexistent for GUI agents. Standard systems pile screenshots into the token window with no compression.

**The gap:** Nobody has published a GUI agent system where the agent explicitly picks between a low-fidelity grayscale wide-field view and a high-fidelity color foveal view -- the two canonical streams of primate vision. Claude's `zoom` is the closest production feature. GUI-Eyes trains the decision via RL. Every existing system treats the screenshot as one monolithic observation -- full RGB, uniform resolution, all at once. The retina does not work that way, and neither should an efficient agent.

---

## The design: LLM chooses how to look

Instead of force-feeding the LLM a screenshot every step, we give it an `observe` tool with three modes. **The three modes map one-to-one onto the three streams the primate retina sends to cortex.** The LLM is the CEO -- it picks which stream it needs.

### Three streams of biological vision, three modes

The human retina outputs roughly three parallel streams. Each is optimized for a different kind of information, and each has a characteristic bandwidth:

| Retinal / cortical stream | What it is good at | Why it is cheap | Agent mode | Returns | Cost |
|---|---|---|---|---|---|
| **Rods -> magnocellular -> dorsal ("where")** | Spatial layout, motion, luminance contrast | Achromatic (1 channel) + low spatial resolution + massive convergence (~100:1 rod-to-ganglion) | `scan` | Low-res grayscale of full screen, with change annotations since last scan | ~200 tokens |
| **Cones -> parvocellular -> ventral ("what")** | Color, fine detail, object identity | High fidelity but only over the ~2 degree foveal window | `focus` | High-res full-color crop at (x, y) | ~150 tokens |
| **VWFA (Visual Word Form Area)** | Symbolic text recognition | Bypasses pixel encoding for words | `read` | Extracted text content at (x, y) | ~40 tokens |

**The efficiency argument is physical, not handwavy.** Grayscale is literally one channel instead of three. A low-resolution sweep has linearly fewer pixels than a full-res frame. A foveal crop touches only ~1/20th of a 1920x1080 screen. Text extraction replaces a 2D pixel array with a 1D token sequence. Every choice mirrors a compression trick the retina already makes, and every choice drops the token cost by an order of magnitude or more against the naive "send one RGB screenshot" baseline.

**The biological argument is not ornament, it is the constraint that forced the design.** An earlier iteration of this work had four modes (`glance`, `read`, `check`, `look`). We collapsed them to three because biology does not provide a fourth stream. There is no brain region that produces a text description of the whole screen (`glance` with text layout output was an LLM-friendly fiction); there is no separate "did my action succeed?" module (that is what the magnocellular motion channel does, once per saccade). Keeping only streams that the retina actually has forces the design to be both simpler and more faithful to the compression strategy it is imitating.

### Why `scan` is grayscale and low-resolution

Rods see in grayscale because rhodopsin is a single photopigment. They are numerous (~120M) and heavily multiplexed: up to 100 rods converge onto a single ganglion cell. This gives them exquisite sensitivity at the cost of resolution and color. The magnocellular ganglion cells they feed have large receptive fields, respond transiently to luminance changes, and carry the dominant motion signal to V1.

For a computer-use agent, this stream answers questions like: "Is there a new window?", "Did a dialog just appear?", "What is the overall layout?", "Is something blinking or animating?" None of these require color, and none require 1920x1080 resolution. A 480x270 grayscale image is enough. The token cost drops by ~5-10x versus a full-color full-res frame, with effectively no loss of spatial-layout and motion information.

Because the magnocellular stream is temporally transient -- it responds to changes -- `scan` also returns a text annotation of regions that changed since the last scan ("Dialog appeared near (1200, 400)"). This is the agent-facing analog of motion/change detection in the dorsal stream, and it makes `scan` a natural choice both for orienting at a new screen and for verifying an action.

### Why `focus` is color, high-resolution, and local

Cones are the opposite design choice: three photopigments (L, M, S) for color, packed densely in the ~2 degree fovea, with near 1:1 cone-to-ganglion ratios. They feed the parvocellular stream, which specializes in high spatial-frequency color-opponent signals -- exactly what object recognition and fine discrimination need.

For a computer-use agent, this stream answers questions like: "What color is this indicator?", "What icon is this?", "What does this chart show?", "Is this pixel-perfect UI element aligned?" These require full fidelity, but only at a point -- the same asymmetry the fovea exploits. A 300x300 full-color crop is about 1/23rd of a full screen. The token cost is several times cheaper than a full RGB screenshot, and the high fidelity at the point of interest is exactly what small-element grounding benchmarks (ScreenSpot-Pro, 0.07% target area) reward.

### Why `read` is its own mode

Reading is a cultural skill, not an evolutionary one -- writing is about 5,000 years old, the fovea is ~500 million. But the brain solved it anyway by dedicating a patch of left fusiform cortex (the Visual Word Form Area, VWFA) to extracting symbolic text from the ventral stream. Functionally, VWFA takes high-res foveal input and emits a string of letters; the downstream language cortex works on strings, not pixels.

For an agent, an OCR or accessibility-API lookup is functionally equivalent to the VWFA: it takes a region of pixels (or in the a11y case, skips the pixels entirely) and returns a string of text. Most human screen interaction is text comprehension -- eye-tracking studies show that over 70% of fixation time is spent on textual elements. Giving the agent a mode that returns text directly, at ~40 tokens per call, skips the parvocellular and VWFA stages altogether. The text *is* the information; the pixels were only ever a carrier.

### Why no `check` mode

The previous design had a dedicated `check` mode for predictive-coding-style action verification: "I expected a save dialog; did it appear?" We dropped it because the biological mechanism it imitated -- change detection in the magnocellular stream after a saccade or motor action -- is already what `scan` does. A `scan` after an action returns a grayscale image plus "Changes since last scan: dialog appeared at (x, y)". The agent reads both and decides whether that matches its expectation, using the same language-reasoning it uses for every other decision. A separate mode would be duplication.

### Why no `glance` mode

The previous design had a `glance` mode that returned a text layout description generated from the a11y tree or from OCR over the whole screen. We dropped it because it inverts the biology in an unhelpful way: biological scene gist is visual (36 ms of low-spatial-frequency magnocellular activity before any object recognition), not a paragraph of sentences. Modern VLMs can read a low-res grayscale image much more naturally than they can consume a synthesized text description of a screen; the translation step was adding latency and information loss for no real benefit. `scan` returns the grayscale image directly, and the LLM's visual encoder does the gist extraction.

### The `observe` tool schema

```python
OBSERVE_TOOL = {
    "name": "observe",
    "description": (
        "Observe the screen. Pick the cheapest mode that gives you "
        "what you need. Three modes, matching the three streams the "
        "retina sends to cortex:\n\n"
        "- 'read': What does the text say here? (~40 tokens, text only). "
        "  Extracts text at coordinates via a11y or OCR. Use for labels, "
        "  menus, error messages, form fields -- most screen interaction.\n"
        "- 'scan': Where are things? What changed? (~200 tokens, grayscale). "
        "  Low-res grayscale of the whole screen plus annotations of "
        "  changes since last scan. Use to orient or to verify actions.\n"
        "- 'focus': What does this look like in detail? (~150 tokens, color). "
        "  High-res full-color crop at coordinates. Use for icons, colors, "
        "  charts, and fine visual layout that grayscale cannot capture."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["read", "scan", "focus"],
            },
            "center_x": {
                "type": "integer",
                "description": "For 'read'/'focus': X center of region"
            },
            "center_y": {
                "type": "integer",
                "description": "For 'read'/'focus': Y center of region"
            },
            "radius": {
                "type": "integer",
                "default": 150,
                "description": "For 'read'/'focus': half-width in pixels"
            }
        },
        "required": ["mode"]
    }
}
```

### The agent loop

```
New screen or navigation
  -> scan (orient: grayscale full-screen; see layout and any changes)
  -> read (extract text near the target element)
  -> act (click, type, scroll, keyboard shortcut)
  -> scan (verify: grayscale full-screen; check what changed near the action point)
    -> change matches expectation: proceed
    -> no change: action failed, try alternative
    -> unexpected change:
         -> read the new text; if enough, proceed
         -> if visual detail needed (icon state, color, chart): focus the region
  -> repeat
```

`focus` (full-color crop) is the exception, not the default. It is called only when `read` and `scan` surface something that text and grayscale cannot resolve. This matches the biological asymmetry: most of the brain's visual bandwidth budget is spent on the magnocellular stream plus text reading; detailed color recognition at the fovea is expensive and used sparingly.

Mapping back to the biology:
- `scan` = rods + magnocellular + dorsal ("where"): wide-field grayscale + motion/change detection
- `focus` = cones + parvocellular + ventral ("what"): local color + high acuity
- `read` = VWFA: symbolic extraction from the ventral stream

---

## V1 implementation

### What to build

One skill named `computer_use`, two perception backends selected by config:

```
fsp/
  computer_use/
    __init__.py       # ComputerUse(perception="standard"|"fsp") -- single entry point
    actions.py        # click, type_text, scroll, keyboard_shortcut (shared by both)
    capture.py        # screen capture, a11y queries, OCR (shared by both)
    memory.py         # Sliding window: keep last N steps as text, drop old images

    perception/
      standard.py     # Full screenshot every step. The control condition.
      fsp.py          # observe tool: read / scan / focus. The treatment.
      observe.py      # Mode dispatch for FSP backend
      read.py         # Text extraction at coordinates via a11y or OCR crop
      scan.py         # Grayscale downsample + inter-frame change detection
      focus.py        # High-res full-color crop with optional a11y annotations

  eval/
    runner.py         # Takes (model, perception, benchmark) -> results
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

**FSP backend:** Model gets `computer_use` with an `observe` sub-tool. Actions and observations are decoupled. The model decides which stream to sample.

### `observe.py` core

```python
from PIL import Image, ImageOps
import imagehash

class Observer:
    def __init__(self, scan_width: int = 480, scan_height: int = 270):
        self.scan_width = scan_width
        self.scan_height = scan_height
        self.prev_scan = None          # last grayscale scan image
        self.prev_hash = None
        self.last_action_point = None  # (x, y) of last click/interaction

    def observe(self, mode: str, **kwargs) -> dict:
        """LLM chooses which retinal stream to sample."""
        if mode == "read":
            return self._read(kwargs["center_x"], kwargs["center_y"],
                              kwargs.get("radius", 150))
        elif mode == "scan":
            return self._scan()
        elif mode == "focus":
            return self._focus(kwargs["center_x"], kwargs["center_y"],
                               kwargs.get("radius", 150))

    def _read(self, cx: int, cy: int, radius: int) -> dict:
        """VWFA: text extraction at a point. No pixels returned."""
        img = capture_screen()
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
            "tokens": max(1, len(text.split()) // 2),
        }

    def _scan(self) -> dict:
        """Rods + magnocellular: grayscale wide-field + change detection."""
        img = capture_screen()
        gray = ImageOps.grayscale(img).resize(
            (self.scan_width, self.scan_height), Image.LANCZOS
        )
        curr_hash = imagehash.phash(img)

        changes_text = ""
        if self.prev_scan is not None and self.prev_hash is not None:
            distance = curr_hash - self.prev_hash
            if distance < 3:
                changes_text = "No visible changes since last scan."
            else:
                regions = diff_regions(self.prev_scan, gray,
                                       anchor=self.last_action_point)
                changes_text = describe_regions(regions)

        self.prev_scan = gray
        self.prev_hash = curr_hash
        return {
            "type": "scan",
            "image": gray,                # 1-channel, ~480x270
            "changes": changes_text,      # "Dialog appeared near (1200, 400)."
            "tokens": (gray.width * gray.height) // 1800 + len(changes_text) // 4,
        }

    def _focus(self, cx: int, cy: int, radius: int) -> dict:
        """Cones + parvocellular: high-res full-color crop at a point."""
        img = capture_screen()
        crop = img.crop((
            max(0, cx - radius), max(0, cy - radius),
            min(img.width, cx + radius), min(img.height, cy + radius)
        ))
        annotations = try_accessibility_in_region(cx, cy, radius)
        return {
            "type": "focus",
            "image": crop,                # full-color, high-res
            "annotations": annotations,
            "tokens": (crop.width * crop.height) // 750,
        }
```

### Memory: vision for the present, text for the past

Following Fara-7B's finding that text history outperforms screenshot history, and JetBrains' finding that simple masking beats summarization:

```python
class SimpleMemory:
    def __init__(self, window: int = 5):
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
You are a desktop automation agent with bio-efficient perception.

Your eyes have three streams, each optimized for a different question:

READ: When you need to know what text says, use mode="read" with
  coordinates. Returns extracted text content (~40 tokens). This is
  the cheapest mode and the most common -- most screen interaction
  is text comprehension.

SCAN: When you arrive at a new screen, or to verify an action,
  use mode="scan". Returns a low-res grayscale view of the whole
  screen, plus a list of regions that changed since your last scan
  (~200 tokens). Use this to orient and to detect motion.

FOCUS: Only when you need color or fine visual detail that grayscale
  cannot capture (icons, colors, charts, precise layout), use
  mode="focus" with coordinates. Returns a full-color high-res crop
  (~150 tokens).

Pick the cheapest mode that answers your question. Prefer "read"
over "scan", and "scan" over "focus". After an action, a single
"scan" both tells you what changed and lets you verify the outcome.
If "scan" says nothing changed, your action likely failed -- try
a different approach rather than repeating.
```

---

## Expected token savings

| Step pattern | Naive (screenshot/step) | FSP (LLM chooses) |
|---|---|---|
| Arrive at new screen, orient | 1,300 | ~200 (scan, grayscale) |
| Read a label, menu, or error message | 1,300 | ~40 (read, text only) |
| Verify action succeeded | 1,300 | ~200 (scan with change annotations) |
| Inspect a visual element (icon, chart) | 1,300 | ~150 (focus, color crop) |
| **Typical 50-step task** | **~65,000** | **~5,000-6,500** |

Estimated **10-13x token reduction** against the naive screenshot-per-step baseline. The breakdown of a typical 50-step task under FSP:

- ~20-25 `read` calls at ~40 tok each (labels, menus, messages): ~1,000 tok
- ~12-15 `scan` calls at ~200 tok each (orient + post-action verify): ~2,500-3,000 tok
- ~8-10 `focus` calls at ~150 tok each (icons, charts, fine visual): ~1,200-1,500 tok
- **Total: ~4,700-5,500 tok.**

Conservative, because it assumes the agent `scan`s frequently. An agent that verifies most actions by `read`ing the expected new text (rather than a full `scan`) can push the total below 3,000 tokens, reaching ~20x reduction. Compatible with existing context compression (Fara-7B's "keep last 3", JetBrains observation masking) for further gains.

**Why the math changed from the earlier 19-33x estimate.** The previous design claimed 19-33x by assuming ~90% of steps would be handled by text-only modes (`read`, `check`, `glance`). The new design is more honest about the physical cost of action verification: you usually need *some* visual evidence that the screen changed, and returning a cheap grayscale image is both more reliable than a text summary and more biologically faithful. We trade ~2x in peak token efficiency for a simpler, more robust three-mode interface that matches the retina's actual architecture.

---

## Evaluation

### Experimental design: 2x2 matrix

The claim is that FSP improves perception efficiency **regardless of the underlying model**. To make this conclusive, we test two models x two perception approaches:

| | Standard CU (full screenshot every step) | FSP (`observe` tool, 3 modes) |
|---|---|---|
| **GPT-5.4** (OpenAI) | Baseline A | Treatment A |
| **Opus 4.6** (Anthropic) | Baseline B | Treatment B |

**The result is conclusive if FSP wins on both rows.** This controls for the objection "maybe your tool just suits one model's training." If both GPT-5.4 and Opus 4.6 benefit, the perception interface itself is better, not model-specific tuning.

### Benchmarks

**Phase 1: ScreenSpot-Pro (grounding accuracy, fast iteration)**
- Static dataset, no VM needed. Quick turnaround.
- Measures: can the agent click the right element?
- Key difficulty: targets occupy only 0.07% of image area -- where `focus` and the zoom/crop approaches shine.
- Run all 4 conditions (2 models x 2 approaches).

**Phase 2: OSWorld (end-to-end task completion)**
- Full desktop tasks in VMs. Slower, more expensive.
- Measures: does the agent complete real tasks?
- The benchmark most cited in observational-efficiency analyses (latency degradation, observation waste, grounding failures).
- Run all 4 conditions.

### Metrics per condition

| Metric | What it tells us |
|--------|-----------------|
| Success rate | FSP must not hurt task completion |
| Total input tokens per task | The core claim: 10-13x reduction |
| Observation tokens only | Isolates perception savings from reasoning tokens |
| Steps to completion | Fewer wasted observation cycles (OSWorld-Human found 1.4-2.7x waste) |
| Mode distribution (FSP only) | Does the agent actually use `read` most? How often does it reach for `focus`? |
| Latency per step | Less context = faster inference |
| Grounding accuracy (ScreenSpot-Pro) | Click precision on small elements |

### Implementation

Both conditions use the same `computer_use` skill. The only variable is the perception backend:

- **`perception="standard"`:** Full screenshot returned after every action. Model sees one `computer_use` tool. Mirrors current Claude/OpenAI production behavior.
- **`perception="fsp"`:** Screenshot captured internally but NOT sent directly. Model sees `computer_use` with `observe` sub-tool (3 modes: `read`, `scan`, `focus`). Actions and observations are decoupled -- the model decides which stream to sample.

Same skill name, same action space, same system prompt (minus perception-specific instructions). The model doesn't know which condition it's in.

---

## What's NOT in V1 (but validated by research for V2+)

| Feature | Research basis | V2 priority |
|---------|---------------|-------------|
| Bottom-up saliency monitor ("attention interrupt") | Itti & Koch saliency; 40% change blindness without attention; fills the top-down-only gap | High -- environmental changes should be able to capture the agent's attention without being asked |
| `find` mode (guided search by target description) | Wolfe's Guided Search; ScreenSeeker's cascaded search (+254%) | High -- replaces blind `elements` dump |
| RL-trained mode selection | GUI-Eyes: 0.6*R_acc + 0.3*R_tool reward; LASER: Monte Carlo preference optimization | High -- agent learns when to scan vs read vs focus |
| Grouped-action execution | OSWorld-Human: agents waste 1.4-2.7x observation cycles | Medium -- multiple actions from one observation |
| Task-conditioned visual encoding | Qwen2.5-VL cross-attention; InternVL prompt-modulated vision | Medium -- encode screenshot differently per task |
| Hierarchical memory consolidation | AgentOCR: RL-selected compression; ACON: NL-optimized guidelines | Medium -- adaptive compression of old steps |
| OmniParser fallback for broken a11y | Screen2AX: only 33% of macOS apps have full a11y; OmniParser v2: 0.6s/frame | Medium -- vision-generated element lists |

---

## The core bet

Every production computer-use system today sends one full-color full-resolution screenshot at every step. The primate retina, solving a structurally identical bandwidth problem, does not. It sends a cheap grayscale wide-field stream (rods / magnocellular), an expensive color foveal stream (cones / parvocellular), and a symbolic text channel (VWFA) -- and it lets higher cortex decide which to attend to.

The bet is: **an LLM that picks among these three streams will perceive more efficiently than one forced to see everything at once, because the retina's compression strategy is the one that actually works on a finite bandwidth budget.** The research agrees from multiple angles: Fara-7B proves text history >> screenshot history, JetBrains proves old observations can be masked at zero cost, OSWorld-Human proves agents take 1.4-2.7x more observation cycles than necessary, and the entire zoom/crop literature proves targeted inspection outperforms full-screen processing. What was missing was a unified interface that packaged the right set of observation modes -- one per retinal stream -- into a simple tool the LLM can drive.

That is `observe` with `read`, `scan`, and `focus`. One text mode, one grayscale wide-field mode, one color foveal mode. Three streams, one decision per step, the same decomposition primate vision has been using for half a billion years.
