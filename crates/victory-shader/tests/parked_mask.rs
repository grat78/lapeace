//! Regression coverage for wave64 scalar values parked in VGPR lanes.

use victory_shader::gfx10::{Fields, Operand};
use victory_shader::ir::{self, PARKED, Reg};

const S_ENDPGM: u32 = 0xBF81_0000;

fn write_lane(scalar: u8, lane: u8) -> [u32; 2] {
    // VOP3 v_writelane_b32 v1, s<scalar>, lane. Inline integer constants
    // 128..=192 encode values 0..=64; the lane selector is source 1.
    [0xD761_0001, (u32::from(128 + lane) << 9) | u32::from(scalar)]
}

fn read_lane(scalar: u8, lane: u8) -> [u32; 2] {
    // VOP3 v_readlane_b32 s<scalar>, v1, lane.
    [
        0xD760_0000 | u32::from(scalar),
        (u32::from(128 + lane) << 9) | 0x101,
    ]
}

fn build_pair(interposed_read: bool) -> ir::Program {
    let mut words = Vec::new();
    // Park the two halves of a wave64 mask in adjacent lanes of v1, with
    // unrelated scalar work between the saves, as the compiler schedules it.
    words.extend(write_lane(0, 3));
    words.push(0xBE82_0387); // s_mov_b32 s2, 7
    words.extend(write_lane(1, 4));

    words.extend(read_lane(0, 3));
    if interposed_read {
        // s1 is still the old scalar value here. Pair-restoring both halves
        // at the first read would make this observe the parked value too soon.
        words.push(0x7E04_0201); // v_mov_b32 v2, s1
    } else {
        words.push(0xBE83_0385); // s_mov_b32 s3, 5
    }
    words.extend(read_lane(1, 4));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ir::build(&code, true, 4).expect("synthetic wave64 program builds")
}

fn scalar(operand: &Operand) -> (u8, u8) {
    match *operand {
        Operand::Scalar { code, dwords } => (code, dwords),
        other => panic!("expected scalar operand, got {other:?}"),
    }
}

#[test]
fn parked_wave64_mask_pair_restores_across_independent_instructions() {
    let program = build_pair(false);
    let instructions = &program.blocks[0].insts;
    let wide_moves: Vec<_> = instructions
        .iter()
        .filter(|inst| inst.name() == "s_mov_b64")
        .collect();
    assert_eq!(wide_moves.len(), 2, "{instructions:#?}");

    let Fields::Sop { dst: Some(save_dst), src: save_src, .. } = &wide_moves[0].fields else {
        panic!("expected the paired save: {:#?}", wide_moves[0]);
    };
    let (parked, width) = scalar(save_dst);
    assert!(parked >= PARKED, "parked register code {parked}");
    assert_eq!(width, 2);
    assert_eq!(scalar(&save_src[0]), (0, 2));

    let Fields::Sop { dst: Some(restore_dst), src: restore_src, .. } = &wide_moves[1].fields else {
        panic!("expected the paired restore: {:#?}", wide_moves[1]);
    };
    assert_eq!(scalar(restore_dst), (0, 2));
    assert_eq!(scalar(&restore_src[0]), (parked, 2));

    // The pair is replaced by two scalar moves; the original lane ops are
    // gone, while unrelated scalar instructions remain in the block.
    assert_eq!(instructions.iter().filter(|i| i.name() == "v_writelane_b32" || i.name() == "v_readlane_b32").count(), 0);
    assert!(instructions.iter().any(|i| i.name() == "s_mov_b32"));
}

#[test]
fn parked_pair_is_not_combined_across_a_read_of_the_second_scalar() {
    let program = build_pair(true);
    let instructions = &program.blocks[0].insts;
    assert!(!instructions.iter().any(|i| i.name() == "s_mov_b64"), "{instructions:#?}");

    let intervening = instructions
        .iter()
        .find(|i| i.name() == "v_mov_b32")
        .expect("interposed scalar read remains");
    assert!(intervening.read(Reg::S(1)).is_some());
}
