use std::collections::HashMap;
use std::hash::Hash;

/// A generic UnionFind (disjoint set) data structure.
/// T must be Clone + Eq + Hash to be used as HashMap keys
#[derive(Debug, Clone)]
pub struct UnionFind<T: Clone + Eq + Hash> {
    parent: HashMap<T, T>,
}

impl<T: Clone + Eq + Hash> UnionFind<T> {
    /// Creates a new empty UnionFind data structure
    pub fn new() -> Self {
        Self { parent: HashMap::new() }
    }

    /// Finds the root/representative element for the set containing x
    /// Uses path compression to optimize future lookups
    pub fn find(&mut self, x: &T) -> T {
        let mut current = x.clone();
        
        // Follow parent pointers until we reach a root
        while let Some(parent) = self.parent.get(&current).cloned() {
            if parent == current {
                break;
            }
            // Path compression - point nodes directly to root
            if let Some(grandparent) = self.parent.get(&parent).cloned() {
                self.parent.insert(current.clone(), grandparent.clone());
                current = grandparent;
            } else {
                current = parent;
            }
        }
        
        // Ensure the element exists in the structure
        self.parent.entry(current.clone()).or_insert(current.clone());
        current
    }

    /// Unions the sets containing elements x and y
    pub fn union(&mut self, x: &T, y: &T) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x != root_y {
            self.parent.insert(root_y, root_x);
        }
    }

    /// Adds multiple pairs of equivalent elements
    pub fn add_equivalences(&mut self, pairs: &[(T, T)]) {
        for (a, b) in pairs {
            self.union(a, b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new();
        
        // Test with integers
        uf.union(&1, &2);
        uf.union(&2, &3);
        
        assert_eq!(uf.find(&1), uf.find(&3));
        assert_eq!(uf.find(&2), uf.find(&3));
        assert_ne!(uf.find(&1), uf.find(&4));
    }

    #[test]
    fn test_union_find_strings() {
        let mut uf = UnionFind::new();
        
        // Test with strings
        uf.union(&"a".to_string(), &"b".to_string());
        uf.union(&"b".to_string(), &"c".to_string());
        
        assert_eq!(uf.find(&"a".to_string()), uf.find(&"c".to_string()));
        assert_eq!(uf.find(&"b".to_string()), uf.find(&"c".to_string()));
        assert_ne!(uf.find(&"a".to_string()), uf.find(&"d".to_string()));
    }

    #[test]
    fn test_add_equivalences() {
        let mut uf = UnionFind::new();
        let pairs = vec![(1, 2), (2, 3), (4, 5)];
        
        uf.add_equivalences(&pairs);
        
        assert_eq!(uf.find(&1), uf.find(&3));
        assert_eq!(uf.find(&4), uf.find(&5));
        assert_ne!(uf.find(&1), uf.find(&4));
    }
} 