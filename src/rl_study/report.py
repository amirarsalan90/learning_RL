"""Dependency-free SVG plots of seed means and one-standard-deviation bands."""

from html import escape
import statistics

from .bandit import BASELINES


def learning_curves_svg(rows):
    metrics = (("expected_reward", "Expected reward", 0.0, 1.0),
               ("optimal_action_probability", "Probability of optimal action", 0.0, 1.0),
               ("entropy", "Policy entropy (nats)", 0.0, 1.15),
               ("gradient_variance", "Gradient variance at current policy", 0.0,
                max(row["gradient_variance"] for row in rows) * 1.15))
    colors = dict(none="#2563eb", running="#ea580c", learned="#16a34a")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700" viewBox="0 0 1000 700">',
           '<rect width="1000" height="700" fill="white"/>',
           '<g font-family="sans-serif" font-size="12" fill="#334155">']
    max_samples = max(row["samples"] for row in rows)
    for index, (metric, title, lower, upper) in enumerate(metrics):
        left, top = 65 + (index % 2) * 490, 55 + (index // 2) * 325
        width, height = 405, 235
        def point(samples, value):
            return f"{left + width * samples / max_samples:.2f},{top + height * (1 - (value - lower) / (upper - lower)):.2f}"
        svg.append(f'<text x="{left}" y="{top - 20}" font-size="16">{escape(title)}</text>')
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            y = top + height * (1 - fraction)
            svg.append(f'<path d="M{left},{y}h{width}" stroke="#e2e8f0"/>')
            svg.append(f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{lower + fraction * (upper - lower):.3f}</text>')
        for fraction in (0, 0.5, 1):
            x = left + width * fraction
            svg.append(f'<text x="{x}" y="{top + height + 18}" text-anchor="middle">{int(max_samples * fraction)}</text>')
        svg.append(f'<text x="{left + width / 2}" y="{top + height + 37}" text-anchor="middle">Sampled actions</text>')
        for method in BASELINES:
            groups = {}
            for row in rows:
                if row["method"] == method:
                    groups.setdefault(row["samples"], []).append(row[metric])
            values = [(s, statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
                      for s, v in sorted(groups.items())]
            band = [point(s, min(upper, m + sd)) for s, m, sd in values]
            band += [point(s, max(lower, m - sd)) for s, m, sd in reversed(values)]
            line = " ".join(point(s, m) for s, m, _ in values)
            svg.append(f'<polygon points="{" ".join(band)}" fill="{colors[method]}" opacity="0.12"/>')
            svg.append(f'<polyline points="{line}" fill="none" stroke="{colors[method]}" stroke-width="2"/>')
    for i, method in enumerate(BASELINES):
        x = 280 + i * 180
        svg.append(f'<rect x="{x}" y="675" width="20" height="4" fill="{colors[method]}"/>')
        svg.append(f'<text x="{x + 28}" y="681">{method}</text>')
    svg.append('</g></svg>')
    return "\n".join(svg)
