//! A minimal SVG line chart: recall@k on the x axis, queries per second on a
//! logarithmic y axis, one series per configuration.

use std::fmt::Write as _;

/// One line on the chart.
pub(crate) struct Series {
    /// Legend text; `\n` starts a second line.
    pub(crate) label: String,
    /// `(recall, qps)` points, drawn in the given order.
    pub(crate) points: Vec<(f64, f64)>,
}

const WIDTH: f64 = 760.0;
const HEIGHT: f64 = 480.0;
const LEFT: f64 = 78.0;
const RIGHT: f64 = 230.0;
const TOP: f64 = 48.0;
const BOTTOM: f64 = 64.0;
/// Color-blind-safe palette (Okabe & Ito, 2008).
const COLORS: [&str; 8] = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442",
];

/// Renders `series` as a standalone SVG document.
pub(crate) fn render(title: &str, subtitle: &str, x_label: &str, series: &[Series]) -> String {
    let points = series.iter().flat_map(|s| s.points.iter());
    let (mut x_min, mut y_min, mut y_max) = (1.0_f64, f64::INFINITY, 0.0_f64);
    for &(x, y) in points {
        x_min = x_min.min(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    // Round the axes outwards: x to a multiple of 0.05, y to whole decades.
    let x_min = ((x_min - 0.02) * 20.0).floor().max(0.0) / 20.0;
    let x_max = 1.0;
    let decade_lo = y_min.max(1.0).log10().floor();
    let decade_hi = y_max.max(10.0).log10().ceil().max(decade_lo + 1.0);
    let plot_w = WIDTH - LEFT - RIGHT;
    let plot_h = HEIGHT - TOP - BOTTOM;
    let sx = |x: f64| LEFT + (x - x_min) / (x_max - x_min) * plot_w;
    let sy = |y: f64| TOP + (decade_hi - y.log10()) / (decade_hi - decade_lo) * plot_h;

    let mut svg = String::new();
    let _ = write!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" font-family="Helvetica, Arial, sans-serif" font-size="12">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{LEFT}" y="22" font-size="15" font-weight="bold" fill="#111">{}</text>
<text x="{LEFT}" y="38" fill="#555">{}</text>
"##,
        escape(title),
        escape(subtitle)
    );
    // Grid and ticks.
    let mut x = x_min;
    while x <= x_max + 1e-9 {
        let px = sx(x);
        let _ = writeln!(
            svg,
            r##"<line x1="{px:.1}" y1="{TOP}" x2="{px:.1}" y2="{:.1}" stroke="#e5e5e5"/><text x="{px:.1}" y="{:.1}" text-anchor="middle" fill="#333">{x:.2}</text>"##,
            TOP + plot_h,
            TOP + plot_h + 18.0
        );
        x += 0.05;
    }
    let mut decade = decade_lo;
    while decade <= decade_hi + 1e-9 {
        let py = sy(10_f64.powf(decade));
        let label = format_qps(10_f64.powf(decade));
        let _ = writeln!(
            svg,
            r##"<line x1="{LEFT}" y1="{py:.1}" x2="{:.1}" y2="{py:.1}" stroke="#e5e5e5"/><text x="{:.1}" y="{:.1}" text-anchor="end" fill="#333">{label}</text>"##,
            LEFT + plot_w,
            LEFT - 8.0,
            py + 4.0
        );
        decade += 1.0;
    }
    let _ = writeln!(
        svg,
        r##"<rect x="{LEFT}" y="{TOP}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#999"/>
<text x="{:.1}" y="{:.1}" text-anchor="middle" fill="#111">{}</text>
<text transform="translate(18 {:.1}) rotate(-90)" text-anchor="middle" fill="#111">queries per second (log scale)</text>"##,
        LEFT + plot_w / 2.0,
        HEIGHT - 18.0,
        escape(x_label),
        TOP + plot_h / 2.0
    );
    // Series and legend.
    for (i, s) in series.iter().enumerate() {
        let color = COLORS[i % COLORS.len()];
        let path: Vec<String> = s
            .points
            .iter()
            .map(|&(x, y)| format!("{:.1},{:.1}", sx(x), sy(y)))
            .collect();
        let _ = writeln!(
            svg,
            r#"<polyline points="{}" fill="none" stroke="{color}" stroke-width="2"/>"#,
            path.join(" ")
        );
        for &(x, y) in &s.points {
            let _ = writeln!(
                svg,
                r#"<circle cx="{:.1}" cy="{:.1}" r="3.5" fill="{color}"/>"#,
                sx(x),
                sy(y)
            );
        }
        legend_entry(
            &mut svg,
            LEFT + plot_w + 14.0,
            TOP + 10.0 + i as f64 * 36.0,
            color,
            &s.label,
        );
    }
    svg.push_str("</svg>\n");
    svg
}

/// A legend swatch at `(x, y)` followed by `label`, one text line per `\n`.
fn legend_entry(svg: &mut String, x: f64, y: f64, color: &str, label: &str) {
    let tx = x + 26.0;
    let mut lines = String::new();
    for (j, line) in label.split('\n').enumerate() {
        let dy = if j == 0 { 0.0 } else { 15.0 };
        let _ = write!(
            lines,
            r#"<tspan x="{tx:.1}" dy="{dy}">{}</tspan>"#,
            escape(line)
        );
    }
    let _ = writeln!(
        svg,
        r##"<line x1="{x:.1}" y1="{y}" x2="{:.1}" y2="{y}" stroke="{color}" stroke-width="2"/><text y="{:.1}" fill="#111">{lines}</text>"##,
        x + 20.0,
        y + 4.0,
    );
}

fn format_qps(v: f64) -> String {
    if v >= 1e6 {
        format!("{}M", v / 1e6)
    } else if v >= 1e3 {
        format!("{}k", v / 1e3)
    } else {
        format!("{v}")
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_points_and_legend() {
        let svg = render(
            "t <1>",
            "s",
            "recall@10",
            &[Series {
                label: "a & b\nsecond line".into(),
                points: vec![(0.8, 5_000.0), (0.99, 800.0)],
            }],
        );
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>\n"));
        assert!(svg.contains("t &lt;1&gt;") && svg.contains("a &amp; b"));
        assert_eq!(svg.matches("<circle").count(), 2);
        assert_eq!(svg.matches("<tspan").count(), 2);
        assert_eq!(format_qps(10_000.0), "10k");
    }
}
