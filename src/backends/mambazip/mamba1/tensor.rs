//! Simple aligned 1D tensor type for SIMD-friendly CPU kernels.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::mem::size_of;
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

const ALIGNMENT: usize = 32;

#[inline]
fn dangling_aligned_f32() -> NonNull<f32> {
    debug_assert_eq!(ALIGNMENT % std::mem::align_of::<f32>(), 0);
    NonNull::new(ALIGNMENT as *mut u8)
        .expect("aligned dangling pointer must be non-null")
        .cast()
}

#[inline]
fn layout_for_f32_elems(len: usize) -> Layout {
    let bytes = len
        .checked_mul(size_of::<f32>())
        .expect("tensor allocation overflow");
    Layout::from_size_align(bytes, ALIGNMENT).expect("invalid layout")
}

#[inline]
fn alloc_f32_buffer(len: usize) -> NonNull<f32> {
    if len == 0 {
        return dangling_aligned_f32();
    }
    let layout = layout_for_f32_elems(len);
    let ptr = unsafe { alloc_zeroed(layout) };
    NonNull::new(ptr)
        .expect("allocation failed")
        .cast()
}

#[inline]
unsafe fn dealloc_f32_buffer(ptr: NonNull<f32>, len: usize) {
    if len == 0 {
        return;
    }
    let layout = layout_for_f32_elems(len);
    unsafe {
        dealloc(ptr.as_ptr() as *mut u8, layout);
    }
}

/// Owned 1D tensor with aligned backing storage.
#[repr(C)]
pub struct Tensor1D {
    data: NonNull<f32>,
    len: usize,
}

impl Tensor1D {
    /// Create a new zero-initialized tensor.
    pub fn zeros(len: usize) -> Self {
        Self {
            data: alloc_f32_buffer(len),
            len,
        }
    }

    /// Create from an existing vector.
    pub fn from_vec(v: Vec<f32>) -> Self {
        let mut t = Self::zeros(v.len());
        t.as_mut_slice().copy_from_slice(&v);
        t
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Immutable raw pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Mutable raw pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_ptr()
    }

    /// Immutable slice over elements.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// Mutable slice over elements.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }

    /// Fill with zeros.
    #[inline]
    pub fn zero(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.data.as_ptr(), 0, self.len);
        }
    }
}

impl Clone for Tensor1D {
    fn clone(&self) -> Self {
        let mut out = Self::zeros(self.len);
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

impl Drop for Tensor1D {
    fn drop(&mut self) {
        unsafe {
            dealloc_f32_buffer(self.data, self.len);
        }
    }
}

unsafe impl Send for Tensor1D {}
unsafe impl Sync for Tensor1D {}

impl Index<usize> for Tensor1D {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len);
        unsafe { &*self.data.as_ptr().add(index) }
    }
}

impl IndexMut<usize> for Tensor1D {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.len);
        unsafe { &mut *self.data.as_ptr().add(index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_len_tensor_uses_aligned_non_allocating_sentinel() {
        let mut t = Tensor1D::zeros(0);
        assert_eq!(t.len(), 0);
        assert!(t.as_slice().is_empty());
        assert!(t.as_mut_slice().is_empty());
        assert_eq!((t.as_ptr() as usize) % ALIGNMENT, 0);
        t.zero();
    }

    #[test]
    fn zero_len_tensor_clone_is_safe() {
        let t = Tensor1D::zeros(0);
        let cloned = t.clone();
        assert_eq!(cloned.len(), 0);
        assert!(cloned.as_slice().is_empty());
        assert_eq!((cloned.as_ptr() as usize) % ALIGNMENT, 0);
    }
}
