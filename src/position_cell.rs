#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ColumnType {
    Input(i32),
    Output,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PositionCell {
    pub row_idx: usize,
    pub wire_type: ColumnType,
}

/// Helper function to convert a PositionCell to a unique identifier
pub fn position_to_id(pos: &PositionCell) -> usize {
    let base = pos.row_idx * 4; // 4 columns per row (3 inputs + 1 output)
    match pos.wire_type {
        ColumnType::Input(0) => base,
        ColumnType::Input(1) => base + 1,
        ColumnType::Input(2) => base + 2,
        ColumnType::Output => base + 3,
        _ => panic!("Invalid column type"),
    }
}