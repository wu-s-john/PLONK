use std::collections::HashMap;

use ark_ff::Field;
use crate::ast::ASTKind;
use crate::execution_trace::{ExecutionTrace, WirePosition};

/// A struct that holds both the offset map and total size of all tables.
/// This helps track where each operation's rows begin in the flattened table
/// and the total number of rows across all operations.
#[derive(Debug, Clone)]
pub struct OffsetTable {
    /// Maps operation kind to their starting row index
    pub offset_map: HashMap<ASTKind, usize>,
    /// Total number of rows across all operations
    pub total_rows: usize,
}

impl OffsetTable {
    /// Builds an OffsetTable from an ExecutionTrace by computing
    /// cumulative row counts across all operations (single/double/triple).
    /// Now keyed by ASTKind instead of string.
    pub fn build<F: Field>(trace: &ExecutionTrace<F>) -> Self {
        let mut offset_map = HashMap::new();
        let mut cumulative = 0;

        // Single-input operations
        for (op_kind, table) in &trace.single_input {
            offset_map.insert(op_kind.clone(), cumulative);
            cumulative += table.rows.len();
        }

        // Double-input operations
        for (op_kind, table) in &trace.double_input {
            offset_map.insert(op_kind.clone(), cumulative);
            cumulative += table.rows.len();
        }

        // Triple-input operations
        for (op_kind, table) in &trace.triple_input {
            offset_map.insert(op_kind.clone(), cumulative);
            cumulative += table.rows.len();
        }

        Self {
            offset_map,
            total_rows: cumulative,
        }
    }

    /// Get the starting row offset for an operation kind
    pub fn get_offset(&self, kind: &ASTKind) -> Option<usize> {
        self.offset_map.get(kind).copied()
    }

    /// Convert a local row index within an operation to a global row index
    pub fn to_global_row(&self, kind: &ASTKind, local_row: usize) -> Option<usize> {
        self.get_offset(kind).map(|offset| offset + local_row)
    }

    /// Convert a WirePosition to a global index in polynomials
    pub fn to_index(&self, pos: &WirePosition) -> usize {
        let base = self.get_offset(&pos.op_kind).unwrap_or(0);
        base + pos.row_idx
    }
}
