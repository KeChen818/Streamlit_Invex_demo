# RISKEX Design & Style Specification

## Overview

RISKEX uses a dark, enterprise-grade aesthetic — high contrast, minimal chrome, data-forward. The visual language draws from modern SaaS dashboards (Linear, Vercel, Raycast) adapted for Streamlit.

---

## Color Palette

All colors are defined as CSS variables on `:root` and should be referenced by name, not by hex, in any new styles.

| Token                | Hex       | Role                                      |
|----------------------|-----------|-------------------------------------------|
| `--riskex-bg`        | `#0f172a` | Page background (deep navy)               |
| `--riskex-panel`     | `#111827` | Secondary panels, hero block              |
| `--riskex-card`      | `#1f2937` | Metric cards, table rows                  |
| `--riskex-border`    | `#334155` | All borders and dividers                  |
| `--riskex-text`      | `#f8fafc` | Primary text (near white)                 |
| `--riskex-muted`     | `#94a3b8` | Labels, captions, secondary text          |
| `--riskex-accent`    | `#f97316` | Highlights, call-to-action (orange)       |

**Background gradient** (main canvas):
```css
background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
```

**Sidebar** background: `#0b1120` (slightly darker than page bg), separated by a `1px solid --riskex-border` right border.

---

## Typography

| Use                  | Size       | Weight | Notes                          |
|----------------------|------------|--------|--------------------------------|
| Page title           | `2rem`     | 800    | Tight letter-spacing (`-0.03em`) |
| Section title        | `1.05rem`  | 700    | Used above inventory/analytics |
| Subtitle / caption   | `0.95rem`  | 400    | Muted color (`--riskex-muted`) |
| Metric label         | `0.8rem`   | 400    | Uppercase feel, muted          |
| Metric value         | `1.6rem`   | 800    | Bold, prominent                |

No custom font family is declared — Streamlit's system font stack is used (typically Inter or SF Pro).

---

## Component Patterns

### Hero Banner (`.riskex-hero`)
- Rounded card (`border-radius: 18px`) with `1px` border
- Slight frosted background (`rgba(17,24,39,0.85)`)
- Contains app title + one-line subtitle
- Used once at the top of the main content area

```html
<div class="riskex-hero">
  <div class="riskex-title">APP NAME — Tagline</div>
  <div class="riskex-subtitle">Short description of the tool.</div>
</div>
```

### Metric Cards (`.metric-card`)
- `border-radius: 16px`, `1px` border, semi-transparent card bg
- Always displayed in a 4-column `st.columns` row
- Structure: small muted label on top, large bold value below

```html
<div class="metric-card">
  <div class="metric-label">LABEL</div>
  <div class="metric-value">42</div>
</div>
```

### Risk Severity Badges
Pill-shaped badges with low-opacity tinted backgrounds. Use inline on table rows or detail panels.

| Severity | Text color  | Background                        | Class           |
|----------|-------------|-----------------------------------|-----------------|
| High     | `#fee2e2`   | `rgba(239, 68, 68, 0.18)` (red)   | `.badge-high`   |
| Medium   | `#fef3c7`   | `rgba(245, 158, 11, 0.18)` (amber)| `.badge-medium` |
| Low      | `#dcfce7`   | `rgba(34, 197, 94, 0.18)` (green) | `.badge-low`    |

All badges: `padding: 0.2rem 0.5rem`, `border-radius: 999px`.

### Chat Messages
- Background: `rgba(31, 41, 55, 0.65)` (semi-transparent card)
- Border: `1px solid rgba(51, 65, 85, 0.8)`
- Border radius: `14px`
- Applied via `div[data-testid="stChatMessage"]`

---

## Layout Principles

- **Page layout**: `wide` mode always — full browser width utilized.
- **Sidebar**: Always expanded by default. Used for filters and AI configuration only.
- **Tabs**: Group major views (`Inventory`, `Analytics`, `AI Chat`) using `st.tabs`. Keep tab count to 3–5.
- **Columns**: Use `st.columns` for metric rows (4 cols) and analytics charts (2 cols). Avoid deeply nested columns.
- **Spacing**: Use `st.divider()` between logical sidebar sections. Use `margin-bottom: 1rem` on cards.

---

## Sidebar Structure (order)

1. App logo / name + caption
2. `st.divider()`
3. Filters (selectboxes, search)
4. `st.divider()`
5. AI Configuration (API key input, confirm button, status)

---

## Status & Feedback Patterns

Use Streamlit's native feedback components — do not build custom ones:

| Situation                        | Component       |
|----------------------------------|-----------------|
| Success / confirmed              | `st.success()`  |
| Informational / loaded from env  | `st.info()`     |
| Soft warning / missing input     | `st.warning()`  |
| Error / invalid key / failure    | `st.error()`    |
| Loading / async work             | `st.spinner()`  |

---

## Do's and Don'ts

**Do:**
- Use CSS variables (`var(--riskex-*)`) for all color references in new styles
- Keep new cards consistent with `border-radius: 14–18px` and `1px` borders
- Use `use_container_width=True` on all charts and dataframes
- Keep sidebar lean — filters + config only, no charts

**Don't:**
- Add bright or saturated colors outside the palette (breaks dark-mode cohesion)
- Use `st.metric()` — the custom `.metric-card` HTML provides consistent styling
- Hardcode hex values in new CSS blocks; always add a token to `:root` first
- Use light backgrounds or white panels anywhere in the layout
