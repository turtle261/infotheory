//! Simple aligned tensor types for SIMD operations.
//!
//! These are minimal, no-frills tensor implementations designed for:
//! - Aligned memory for AVX2 (32-byte alignment)
//! - Direct access to underlying data
//! - Zero-copy views for weights

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

/// 32-byte alignment for AVX2
const ALIGNMENT: usize = 32;

/// Owned 1D tensor with aligned memory.
#[repr(C)]
pub struct Tensor1D {
    data: NonNull<f32>,
    len: usize,
}

impl Tensor1D {
    /// Create a new zero-initialized tensor.
    pub fn zeros(len: usize) -> Self {
        let layout = Layout::from_size_align(len * 4, ALIGNMENT).expect("Invalid layout");

        let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        let data = NonNull::new(ptr).expect("Allocation failed");

        Self { data, len }
    }

    /// Create from an existing `Vec<f32>` (may copy if not aligned).
    pub fn from_vec(v: Vec<f32>) -> Self {
        let mut t = Self::zeros(v.len());
        t.as_mut_slice().copy_from_slice(&v);
        t
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

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

    /// Copy from another tensor.
    #[inline]
    pub fn copy_from(&mut self, other: &Tensor1D) {
        debug_assert_eq!(self.len, other.len);
        self.as_mut_slice().copy_from_slice(other.as_slice());
    }

    /// Copy from slice.
    #[inline]
    pub fn copy_from_slice(&mut self, slice: &[f32]) {
        debug_assert_eq!(self.len, slice.len());
        self.as_mut_slice().copy_from_slice(slice);
    }
}

impl Clone for Tensor1D {
    fn clone(&self) -> Self {
        let mut new = Self::zeros(self.len);
        new.as_mut_slice().copy_from_slice(self.as_slice());
        new
    }
}

impl Drop for Tensor1D {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len * 4, ALIGNMENT).expect("Invalid layout");
        unsafe {
            dealloc(self.data.as_ptr() as *mut u8, layout);
        }
    }
}

// Safety: Tensor1D owns its data
unsafe impl Send for Tensor1D {}
unsafe impl Sync for Tensor1D {}

impl Index<usize> for Tensor1D {
    type Output = f32;

    #[inline]
    fn index(&self, i: usize) -> &f32 {
        debug_assert!(i < self.len);
        unsafe { &*self.data.as_ptr().add(i) }
    }
}

impl IndexMut<usize> for Tensor1D {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        debug_assert!(i < self.len);
        unsafe { &mut *self.data.as_ptr().add(i) }
    }
}

/// Owned 2D tensor with aligned memory (row-major).
#[repr(C)]
pub struct Tensor2D {
    data: NonNull<f32>,
    rows: usize,
    cols: usize,
    stride: usize, // stride in elements (rounded up for alignment)
}

impl Tensor2D {
    /// Create a new zero-initialized 2D tensor.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        // Pad cols to 8 for AVX2 alignment
        let stride = (cols + 7) & !7;
        let total = rows * stride;

        let layout = Layout::from_size_align(total * 4, ALIGNMENT).expect("Invalid layout");

        let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        let data = NonNull::new(ptr).expect("Allocation failed");

        Self {
            data,
            rows,
            cols,
            stride,
        }
    }

    /// Create from Vec with shape.
    pub fn from_vec(v: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(v.len(), rows * cols);
        let mut t = Self::zeros(rows, cols);

        // Copy row by row to handle stride
        for r in 0..rows {
            let src_start = r * cols;
            let src_end = src_start + cols;
            t.row_mut(r).copy_from_slice(&v[src_start..src_end]);
        }
        t
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_ptr()
    }

    /// Get a row slice.
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        debug_assert!(r < self.rows);
        unsafe {
            let ptr = self.data.as_ptr().add(r * self.stride);
            std::slice::from_raw_parts(ptr, self.cols)
        }
    }

    /// Get a mutable row slice.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        debug_assert!(r < self.rows);
        unsafe {
            let ptr = self.data.as_ptr().add(r * self.stride);
            std::slice::from_raw_parts_mut(ptr, self.cols)
        }
    }

    /// Get raw row pointer (includes stride padding).
    #[inline]
    pub fn row_ptr(&self, r: usize) -> *const f32 {
        debug_assert!(r < self.rows);
        unsafe { self.data.as_ptr().add(r * self.stride) }
    }

    /// Get raw mutable row pointer.
    #[inline]
    pub fn row_ptr_mut(&mut self, r: usize) -> *mut f32 {
        debug_assert!(r < self.rows);
        unsafe { self.data.as_ptr().add(r * self.stride) }
    }

    /// Fill with zeros.
    pub fn zero(&mut self) {
        let total = self.rows * self.stride;
        unsafe {
            std::ptr::write_bytes(self.data.as_ptr(), 0, total);
        }
    }
}

impl Clone for Tensor2D {
    fn clone(&self) -> Self {
        let total = self.rows * self.stride;
        let layout = Layout::from_size_align(total * 4, ALIGNMENT).expect("Invalid layout");

        let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        let data = NonNull::new(ptr).expect("Allocation failed");

        unsafe {
            std::ptr::copy_nonoverlapping(self.data.as_ptr(), ptr, total);
        }

        Self {
            data,
            rows: self.rows,
            cols: self.cols,
            stride: self.stride,
        }
    }
}

impl Drop for Tensor2D {
    fn drop(&mut self) {
        let total = self.rows * self.stride;
        let layout = Layout::from_size_align(total * 4, ALIGNMENT).expect("Invalid layout");
        unsafe {
            dealloc(self.data.as_ptr() as *mut u8, layout);
        }
    }
}

// Safety: Tensor2D owns its data
unsafe impl Send for Tensor2D {}
unsafe impl Sync for Tensor2D {}

/// View into external f32 data (for weights).
#[derive(Clone, Copy)]
pub struct TensorView1D<'a> {
    data: &'a [f32],
}

impl<'a> TensorView1D<'a> {
    #[inline]
    pub fn new(data: &'a [f32]) -> Self {
        Self { data }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        self.data
    }
}

impl<'a> Index<usize> for TensorView1D<'a> {
    type Output = f32;

    #[inline]
    fn index(&self, i: usize) -> &f32 {
        &self.data[i]
    }
}

/// View into external f32 data (for weights), row-major.
#[derive(Clone, Copy)]
pub struct TensorView2D<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
}

impl<'a> TensorView2D<'a> {
    #[inline]
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        debug_assert!(r < self.rows);
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    #[inline]
    pub fn row_ptr(&self, r: usize) -> *const f32 {
        debug_assert!(r < self.rows);
        unsafe { self.data.as_ptr().add(r * self.cols) }
    }

    /// Transpose view (returns new TensorView with swapped dims).
    /// Note: This is a logical transpose - data is still row-major of original.
    /// Use only for matmuls that handle transposed right operand.
    pub fn t(&self) -> TransposedView2D<'a> {
        TransposedView2D {
            data: self.data,
            rows: self.cols, // swapped
            cols: self.rows, // swapped
            orig_cols: self.cols,
        }
    }
}

/// Transposed view (for efficient transpose-multiply).
#[derive(Clone, Copy)]
pub struct TransposedView2D<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
    orig_cols: usize,
}

impl<'a> TransposedView2D<'a> {
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get element at (r, c) in transposed view.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        // In transposed view, (r, c) maps to original (c, r)
        self.data[c * self.orig_cols + r]
    }

    /// Get original row (which is a column in transposed view).
    #[inline]
    pub fn orig_row(&self, r: usize) -> &[f32] {
        let start = r * self.orig_cols;
        &self.data[start..start + self.orig_cols]
    }
}
