//! Integration tests for importing PDF patterns and SVGs, altering their splines
//! and lines, classifying cutting lines vs seam allowances, and meshing/creating
//! projects from them.

use cad_core::{HandleKind, HandleRef, Segment};
use glam::Vec2;
use io_cad::{LineRole, PdfOptions, SvgOptions, import_pattern_pdf, import_pattern_svg};
use topology_sync::{GarmentTopology, TopologySync};

#[test]
fn test_pdf_with_splines_and_lines_creates_alterable_project() {
    // A PDF pattern containing lines and cubic splines, not made with Inkscape.
    // 0 0 -> 100 0 (line)
    // 100 0 -> 120 50 -> 120 80 -> 100 100 (cubic spline armhole/curve)
    // 100 100 -> 0 100 (line)
    // 0 100 -> 0 0 (line)
    let pdf_content = b"
        q
        1 0 0 1 0 0 cm
        0 0 m
        100 0 l
        120 50 120 80 100 100 c
        0 100 l
        h S
        Q
    ";
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(pdf_content, 6);
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.5\n1 0 obj\n<< /Length ");
    pdf.extend_from_slice(compressed.len().to_string().as_bytes());
    pdf.extend_from_slice(b" /Filter /FlateDecode >>\nstream\n");
    pdf.extend_from_slice(&compressed);
    pdf.extend_from_slice(b"\nendstream\nendobj\ntrailer\n<<>>\n%%EOF");

    let opts = PdfOptions {
        scale: 1.0,
        default_allowance_mm: 10.0,
    };
    let imported = import_pattern_pdf(&pdf, &opts).expect("PDF import should succeed");

    assert_eq!(imported.pieces.len(), 1);
    let candidate = &imported.pieces[0];
    assert_eq!(candidate.loops.len(), 1);

    let outline = &candidate.loops[0].outline;
    assert_eq!(outline.edge_count(), 4);
    assert_eq!(outline.segments()[0], Segment::Line);
    assert!(matches!(outline.segments()[1], Segment::Cubic { .. }));
    assert_eq!(outline.segments()[2], Segment::Line);
    assert_eq!(outline.segments()[3], Segment::Line);

    // Create a project document from the imported pattern
    let mut doc = imported.to_document();
    assert_eq!(doc.pieces().len(), 1);
    let piece_id = doc.pieces()[0].id;

    // Directly alter the spline by moving its anchor
    let handle = HandleRef {
        piece: piece_id,
        kind: HandleKind::Anchor(2),
    };
    let initial_pos = doc.handle_position(handle).unwrap();
    let new_pos = initial_pos + Vec2::new(15.0, 5.0);
    assert!(
        doc.piece_mut(piece_id)
            .unwrap()
            .outline
            .set_handle_position(handle.kind, new_pos)
    );
    assert_eq!(doc.handle_position(handle).unwrap(), new_pos);

    // Verify that topology / cloth meshing works seamlessly on this imported and altered piece
    let mut topology = GarmentTopology::default();
    topology.rebuild(&doc);
    let mesh = topology.mesh();
    assert!(mesh.triangle_count() > 0);
    assert!(mesh.vertex_count() > 0);
}

#[test]
fn test_pdf_from_non_inkscape_sources_with_v_y_and_re_operators() {
    // Test raw PDF stream operators from commercial CAD/Illustrator:
    // v: curveto with first control point at current point
    // y: curveto with second control point at endpoint
    // re: rectangle
    let stream = b"
        0 0 100 100 re W n
        10 10 m
        50 10 l
        60 30 50 50 v
        40 70 10 50 y
        h S
    ";
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n1 0 obj\n<< /Length 200 >>\nstream\n");
    pdf.extend_from_slice(stream);
    pdf.extend_from_slice(b"\nendstream\nendobj\ntrailer\n<<>>\n%%EOF");

    let opts = PdfOptions {
        scale: 1.0,
        default_allowance_mm: 8.0,
    };
    let imported = import_pattern_pdf(&pdf, &opts).unwrap();
    assert!(imported.pieces.len() >= 1);
    let p = &imported.pieces[0];
    // Must contain cubic beziers generated from v and y operators in its loops
    assert!(p.loops.iter().any(|lp| {
        lp.outline
            .segments()
            .iter()
            .any(|s| matches!(s, Segment::Cubic { .. }))
    }));
}

#[test]
fn test_svg_with_cubic_and_quadratic_splines_and_lines() {
    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="500mm" height="500mm">
        <!-- Piece with cubic bezier (C), smooth cubic (S), quadratic (Q), and lines (L) -->
        <path d="M 50 50 L 250 50 C 270 100 270 150 250 200 S 230 300 250 350 Q 150 400 50 350 Z" />
    </svg>"#;

    let opts = SvgOptions {
        scale: Some(1.0),
        flip_y: false,
        default_allowance_mm: 12.0,
    };
    let imported = import_pattern_svg(svg.as_bytes(), &opts).unwrap();

    assert_eq!(imported.pieces.len(), 1);
    let piece = &imported.pieces[0];
    let outline = &piece.loops[0].outline;

    // Verify segments contain Line, Cubic, and Quadratic
    assert!(
        outline
            .segments()
            .iter()
            .any(|s| matches!(s, Segment::Line))
    );
    assert!(
        outline
            .segments()
            .iter()
            .any(|s| matches!(s, Segment::Cubic { .. }))
    );
    assert!(
        outline
            .segments()
            .iter()
            .any(|s| matches!(s, Segment::Quadratic { .. }))
    );

    let doc = imported.to_document();
    let mut topology = GarmentTopology::default();
    topology.rebuild(&doc);
    assert!(topology.mesh().triangle_count() > 0);
}

#[test]
fn test_lines_selectable_as_cutting_line_or_seam_allowance_and_role_swapping() {
    // Pattern with both a cut line (outer 140x140) and a seam line (inner 120x120)
    let svg = r#"<svg viewBox="0 0 400 400" width="400mm" height="400mm">
        <!-- Outer cutting line -->
        <path d="M 0 0 L 140 0 L 140 140 L 0 140 Z" />
        <!-- Inner sew line (10mm offset inside) -->
        <path d="M 10 10 L 130 10 L 130 130 L 10 130 Z" />
    </svg>"#;

    let opts = SvgOptions {
        scale: Some(1.0),
        flip_y: false,
        default_allowance_mm: 10.0,
    };
    let mut imported = import_pattern_svg(svg.as_bytes(), &opts).unwrap();

    assert_eq!(imported.pieces.len(), 1);
    assert_eq!(imported.pieces[0].loops.len(), 2);

    // Automatic detection classified outer as CuttingLine and inner as SeamAllowance
    assert_eq!(
        imported.pieces[0].cut_loop().unwrap().role,
        LineRole::CuttingLine
    );
    assert_eq!(
        imported.pieces[0].sew_loop().unwrap().role,
        LineRole::SeamAllowance
    );
    assert!((imported.pieces[0].seam_allowance_mm - 10.0).abs() < 1.0);

    // Build document: the sew loop (10..130) is the outline
    let doc1 = imported.to_document();
    let p1 = &doc1.pieces()[0];
    assert_eq!(p1.outline.points()[0].position, Vec2::new(10.0, 10.0));
    assert_eq!(p1.seam_allowance.width(0), 10.0);

    // Cutting line outline generated from sew line + seam allowance matches outer boundary (0..140)
    let cut_outline = p1.seam_allowance_outline(1.0).unwrap();
    assert_eq!(cut_outline[0], Vec2::new(0.0, 0.0));
    assert_eq!(cut_outline[1], Vec2::new(140.0, 0.0));

    // User edits stuff if something was wrong: SWAP ROLES
    imported.pieces[0].swap_roles();
    assert_eq!(imported.pieces[0].loops[0].role, LineRole::SeamAllowance);
    assert_eq!(imported.pieces[0].loops[1].role, LineRole::CuttingLine);

    // Now the outer loop is the sew line
    let doc2 = imported.to_document();
    let p2 = &doc2.pieces()[0];
    assert_eq!(p2.outline.points()[0].position, Vec2::new(0.0, 0.0));

    // Also test converting in-place on PatternPiece
    let mut p_copy = p1.clone();
    // Convert from sew line to cutting line (offset outward by 10mm)
    p_copy.convert_sew_line_to_cutting_line();
    assert_eq!(p_copy.outline.points()[0].position, Vec2::new(0.0, 0.0));

    // Convert back from cutting line to sew line (offset inward by 10mm)
    p_copy.convert_cutting_line_to_sew_line();
    assert_eq!(p_copy.outline.points()[0].position, Vec2::new(10.0, 10.0));
}
