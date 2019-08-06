#![feature(test)]
extern crate test;
use std::cmp;
use std::mem::{self, MaybeUninit};
use std::ptr;
use test::{black_box, Bencher};

/// Rotates the range `[mid-left, mid+right)` such that the element at `mid` becomes the first
/// element. Equivalently, rotates the range `left` elements to the left or `right` elements to the
/// right.
///
/// # Safety
///
/// The specified range must be valid for reading and writing.
///
/// # Algorithm
///
/// Algorithm 1 is used for small values of `left + right` or for large `T`. The elements are moved
/// into their final positions one at a time starting at `mid - left` and advancing by `right` steps
/// modulo `left + right`, such that only one temporary is needed. Eventually, we arrive back at
/// `mid - left`. However, if `gcd(left + right, right)` is not 1, the above steps skipped over
/// elements. For example:
/// ```text
/// left = 10, right = 6
/// the `^` indicates an element in its final place
/// 6 7 8 9 10 11 12 13 14 15 . 0 1 2 3 4 5
/// after using one step of the above algorithm (The X will be overwritten at the end of the round,
/// and 12 is stored in a temporary):
/// X 7 8 9 10 11 6 13 14 15 . 0 1 2 3 4 5
///               ^
/// after using another step (now 2 is in the temporary):
/// X 7 8 9 10 11 6 13 14 15 . 0 1 12 3 4 5
///               ^                 ^
/// after the third step (the steps wrap around, and 8 is in the temporary):
/// X 7 2 9 10 11 6 13 14 15 . 0 1 12 3 4 5
///     ^         ^                 ^
/// after 7 more steps, the round ends with the temporary 0 getting put in the X:
/// 0 7 2 9 4 11 6 13 8 15 . 10 1 12 3 14 5
/// ^   ^   ^    ^    ^       ^    ^    ^
/// ```
/// Fortunately, the number of skipped over elements between finalized elements is always equal, so
/// we can just offset our starting position and do more rounds (the total number of rounds is the
/// `gcd(left + right, right)` value). The end result is that all elements are finalized once and
/// only once.
///
/// Algorithm 2 is used if `left + right` is large but `min(left, right)` is small enough to
/// fit onto a stack buffer. The `min(left, right)` elements are copied onto the buffer, `memmove`
/// is applied to the others, and the ones on the buffer are moved back into the hole on the
/// opposite side of where they originated.
///
/// Algorithms that can be vectorized outperform the above once `left + right` becomes large enough.
/// Algorithm 1 can be vectorized by chunking and performing many rounds at once, but there are too
/// few rounds on average until `left + right` is enormous, and the worst case of a single
/// round is always there. Instead, algorithm 3 utilizes repeated swapping of
/// `min(left, right)` elements until a smaller rotate problem is left.
///
/// ```text
/// left = 11, right = 4
/// [4 5 6 7 8 9 10 11 12 13 14 . 0 1 2 3]
///                  ^  ^  ^  ^   ^ ^ ^ ^ swapping the right most elements with elements to the left
/// [4 5 6 7 8 9 10 . 0 1 2 3] 11 12 13 14
///        ^ ^ ^  ^   ^ ^ ^ ^ swapping these
/// [4 5 6 . 0 1 2 3] 7 8 9 10 11 12 13 14
/// we cannot swap any more, but a smaller rotation problem is left to solve
/// ```
/// when `left < right` the swapping happens from the left instead.
pub unsafe fn ptr_rotate<T>(mut left: usize, mut mid: *mut T, mut right: usize) {
    type BufType = [usize; 32];
    if mem::size_of::<T>() == 0 {
        return;
    }
    loop {
        // N.B. the below algorithms can fail if these cases are not checked
        if (right == 0) || (left == 0) {
            return;
        }
        if (left + right < 24) || (mem::size_of::<T>() > mem::size_of::<[usize; 4]>()) {
            // Algorithm 1
            // Microbenchmarks indicate that the average performance for random shifts is better all
            // the way until about `left + right == 32`, but the worst case performance breaks even
            // around 16. 24 was chosen as middle ground. If the size of `T` is larger than 4
            // `usize`s, this algorithm also outperforms other algorithms.
            let x = mid.sub(left);
            // beginning of first round
            let mut tmp: T = x.read();
            let mut i = right;
            // `gcd` can be found before hand by calculating `gcd(left + right, right)`,
            // but it is faster to do one loop which calculates the gcd as a side effect, then
            // doing the rest of the chunk
            let mut gcd = right;
            // benchmarks reveal that it is faster to swap temporaries all the way through instead
            // of reading one temporary once, copying backwards, and then writing that temporary at
            // the very end. This is possibly due to the fact that swapping or replacing temporaries
            // uses only one memory address in the loop instead of needing to manage two.
            loop {
                tmp = x.add(i).replace(tmp);
                // instead of incrementing `i` and then checking if it is outside the bounds, we
                // check if `i` will go outside the bounds on the next increment. This prevents
                // any wrapping of pointers or `usize`.
                if i >= left {
                    i -= left;
                    if i == 0 {
                        // end of first round
                        x.write(tmp);
                        break;
                    }
                    // this conditional must be here if `left + right >= 15`
                    if i < gcd {
                        gcd = i;
                    }
                } else {
                    i += right;
                }
            }
            // finish the chunk with more rounds
            for start in 1..gcd {
                tmp = x.add(start).read();
                i = start + right;
                loop {
                    tmp = x.add(i).replace(tmp);
                    if i >= left {
                        i -= left;
                        if i == start {
                            x.add(start).write(tmp);
                            break;
                        }
                    } else {
                        i += right;
                    }
                }
            }
            return;
        // `T` is not a zero-sized type, so it's okay to divide by its size.
        } else if cmp::min(left, right) <= mem::size_of::<BufType>() / mem::size_of::<T>() {
            // Algorithm 2
            // The `[T; 0]` here is to ensure this is appropriately aligned for T
            let mut rawarray = MaybeUninit::<(BufType, [T; 0])>::uninit();
            let buf = rawarray.as_mut_ptr() as *mut T;
            let dim = mid.sub(left).add(right);
            if left <= right {
                ptr::copy_nonoverlapping(mid.sub(left), buf, left);
                ptr::copy(mid, mid.sub(left), right);
                ptr::copy_nonoverlapping(buf, dim, left);
            } else {
                ptr::copy_nonoverlapping(mid, buf, right);
                ptr::copy(mid.sub(left), dim, left);
                ptr::copy_nonoverlapping(buf, mid.sub(left), right);
            }
            return;
        } else if left >= right {
            // Algorithm 3
            // There is an alternate way of swapping that involves finding where the last swap
            // of this algorithm would be, and swapping using that last chunk instead of swapping
            // adjacent chunks like this algorithm is doing, but this way is still faster.
            loop {
                ptr::swap_nonoverlapping(mid.sub(right), mid, right);
                mid = mid.sub(right);
                left -= right;
                if left < right {
                    break;
                }
            }
        } else {
            // Algorithm 3, `left < right`
            loop {
                ptr::swap_nonoverlapping(mid.sub(left), mid, left);
                mid = mid.add(left);
                right -= left;
                if right < left {
                    break;
                }
            }
        }
    }
}

// this is for direct inclusion into the slice test code
/*
#[test]
fn brute_force_rotate_test() {
    // In case of edge cases involving multiple algorithms
    #[cfg(not(miri))]
    let n = 300;
    // `ptr_rotate` covers so many kinds of pointer usage,
    // that this is just a good test for pointers in general
    #[cfg(miri)]
    let n = 30;
    for len in 0..n {
        for s in 0..len {
            let mut v = Vec::with_capacity(len);
            for i in 0..len {
                v.push(i);
            }
            v[..].rotate_right(s);
            for i in 0..v.len() {
                assert_eq!(v[i], v.len().wrapping_add(i.wrapping_sub(s)) % v.len());
            }
        }
    }
}
*/

pub fn rotate_right(x: &mut [usize], k: usize) {
    assert!(k <= x.len());
    let mid = x.len() - k;

    unsafe {
        let p = x.as_mut_ptr();
        ptr_rotate(mid, p.add(mid), k);
    }
}

// set up a vector that can be checked with the `check` function after the `rotate_right` function
// is called on it.
fn setup(size: usize) -> Vec<usize> {
    let mut v = Vec::with_capacity(size);
    for i in 0..size {
        v.push(i);
    }
    v
}

// assert that the output of a rotate right function is correct
fn check(x: &mut [usize], s: usize) {
    for i in 0..x.len() {
        assert_eq!(x[i], x.len().wrapping_add(i.wrapping_sub(s)) % x.len());
    }
}

#[cfg(test)]
#[test]
fn brute_force_rotate_test() {
    // just in case there is some wierd edge case around 16 * 16
    const N: usize = 300;
    for len in 0..N {
        for s in 0..len {
            let y = &mut setup(len)[..];
            rotate_right(y, s);
            check(y, s);
        }
    }
}

macro_rules! rotate_bench {
    ($bench_name_old:ident, $bench_name_new:ident, $len:expr, $s:expr) => {
        #[bench]
        fn $bench_name_old(b: &mut Bencher) {
            let mut x = black_box(setup($len));
            b.iter(|| {
                x[..].rotate_right($s);
                black_box(x[0].clone())
            })
        }

        #[bench]
        fn $bench_name_new(b: &mut Bencher) {
            let mut x = black_box(setup($len));
            b.iter(|| {
                rotate_right(&mut x[..], $s);
                black_box(x[0].clone())
            })
        }
    };
}

macro_rules! rotate {
    ($fn_old:ident, $fn_new:ident, $n:expr, $mapper:expr) => {
        #[bench]
        fn $fn_old(b: &mut Bencher) {
            let mut x = (0usize..$n).map(&$mapper).collect::<Vec<_>>();
            b.iter(|| {
                for s in 0..x.len() {
                    x[..].rotate_right(s);
                }
                black_box(x[0].clone())
            })
        }

        #[bench]
        fn $fn_new(b: &mut Bencher) {
            let mut x = (0usize..$n).map(&$mapper).collect::<Vec<_>>();
            b.iter(|| {
                for s in 0..x.len() {
                    unsafe { ptr_rotate(x.len().wrapping_sub(s), x.as_mut_ptr().add(x.len().wrapping_sub(s)), s); }
                }
                black_box(x[0].clone())
            })
        }
    }
}

#[derive(Clone)]
struct Rgb(u8,u8,u8);

#[cfg(test)]
mod benches {
    use super::*;

    // small values
    rotate_bench!(x10s3_old, x10s3_new, 10, 3);
    rotate_bench!(x7s3_old, x7s3_new, 7, 3);
    rotate_bench!(x7s0_old, x7s0_new, 7, 0);
    rotate_bench!(x15s3_old, x15s3_new, 15, 3);
    rotate_bench!(x14s7_old, x14s7_new, 14, 7);
    rotate_bench!(x15s13_old, x15s13_new, 15, 13);

    // at the cutoff point between algorithm 1 and the others
    rotate_bench!(x25s24_old, x25s24_new, 25, 24);
    rotate_bench!(x25s1_old, x25s1_new, 25, 1);
    rotate_bench!(x25s5_old, x25s5_new, 25, 5);
    rotate_bench!(x24s12_old, x24s12_new, 24, 12);
    rotate_bench!(x24s20_old, x24s20_new, 24, 20);
    rotate_bench!(x24s1_old, x24s1_new, 24, 1);
    rotate_bench!(x24s23_old, x24s23_new, 24, 23);

    // large values.
    rotate_bench!(x63s40_old, x63s40_new, 63, 40);
    rotate_bench!(x200s40_old, x200s40_new, 200, 40);
    rotate_bench!(x256s17_old, x256s17_new, 256, 17);
    rotate_bench!(x256s0_old, x256s0_new, 256, 0);
    rotate_bench!(x256s1_old, x256s1_new, 256, 1);
    rotate_bench!(x256s3_old, x256s3_new, 256, 3);
    rotate_bench!(x1024s32_old, x1024s32_new, 1024, 32);
    rotate_bench!(x1024s128_old, x1024s128_new, 1024, 128);
    rotate_bench!(x1031s149_old, x1031s149_new, 1031, 149);
    rotate_bench!(x1024s256_old, x1024s256_new, 1024, 256);
    rotate_bench!(x1024s512_old, x1024s512_new, 1024, 512);
    rotate_bench!(x1024s768_old, x1024s768_new, 1024, 768);
    rotate_bench!(x1024s992_old, x1024s992_new, 1024, 992);

    rotate!(rotate_u8_old, rotate_u8_new, 32, |i| i as u8);
    rotate!(rotate_rgb_old, rotate_rgb_new, 32, |i| Rgb(i as u8, (i as u8).wrapping_add(7), (i as u8).wrapping_add(42)));
    rotate!(rotate_usize_old, rotate_usize_new, 32, |i| i);
    rotate!(rotate_16_usize_4_old, rotate_16_usize_4_new, 16, |i| [i; 4]);
    rotate!(rotate_16_usize_5_old, rotate_16_usize_5_new, 16, |i| [i; 5]);
    rotate!(rotate_64_usize_4_old, rotate_64_usize_4_new, 64, |i| [i; 4]);
    rotate!(rotate_64_usize_5_old, rotate_64_usize_5_new, 64, |i| [i; 5]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// below this point is a bunch of old alternative algorithms I made during the production of the
// above, in case someone finds it useful
////////////////////////////////////////////////////////////////////////////////////////////////////

pub fn main() {
}
/*
//A rotate left function that assumes `(1 <= s < x.len()) && (2 <= x.len() < 16)`
unsafe fn slice_rotate_right(x: &mut [u64], s: usize) {
    //in order to avoid allocation, we have to "leapfrog" and wrap around the indexing.
    let mut start = 0;
    let mut lowest_nonstart = s;
    loop {
        let mut tmp0 = *x.get_unchecked(start);
        let mut tmp1;
        let mut i = start + s;
        loop {
            tmp1 = *x.get_unchecked(i);
            *x.get_unchecked_mut(i) = tmp0;
            i += s;
            if i >= x.len() {
                i -= x.len();
                if i == start {
                    *x.get_unchecked_mut(i) = tmp1;
                    break
                }
                //we cannot just have a `lowest_nonstart = i` here without a conditional `i < lowest_nonstart`, unless x.len() is below 16
                //(shortest counterexample is x.len() = 16 and s = 6)
                if i < lowest_nonstart {
                    lowest_nonstart = i;
                }
            }
            //Repeat with swapped temporaries in case the compiler cannot do it.
            //We do not need a second indexing variable to handle the source
            //digit, because that source variable is already stored in tmp0 or
            //tmp1.
            tmp0 = *x.get_unchecked(i);
            *x.get_unchecked_mut(i) = tmp1;
            i += s;
            if i >= x.len() {
                i -= x.len();
                if i == start {
                    *x.get_unchecked_mut(i) = tmp0;
                    break
                }
                if i < lowest_nonstart {
                    lowest_nonstart = i;
                }
            }
        }
        //the leapfrog wraps around until it arrives back where it started, but
        //not all digits have been shifted the first time if `(x.len() % digits) == 0`.
        start += 1;
        if start == lowest_nonstart {return}
    }
}*/
//Using forward replacement with 2 indexes and 1 temporary
/*
//A rotate left function that assumes `(1 <= s < x.len()) && (2 <= x.len() < 16)`
unsafe fn slice_rotate_right(x: &mut [u64], s: usize) {
    let wrap_point = x.len() - s;
    //in order to avoid allocation, we have to "leapfrog" and wrap around the indexing.
    let mut lowest_nonstart = s;
    let mut tmp = *x.get_unchecked(0);
    let mut i0 = 0;
    let mut i1 = s;
    loop {
        tmp = std::mem::replace(x.get_unchecked_mut(i1), tmp);
        i0 = i1;
        if i1 >= wrap_point {
            i1 -= wrap_point;
            if i1 == 0 {
                *x.get_unchecked_mut(i1) = tmp;
                break
            }
            //we cannot just have a `lowest_nonstart = i` here without a conditional `i < lowest_nonstart`, unless x.len() is below 16
            //(shortest counterexample is x.len() = 16 and s = 6)
            if i1 < lowest_nonstart {
                lowest_nonstart = i1;
            }
        } else {
            i1 += s;
        }
    }
    //the leapfrog wraps around until it arrives back where it started, but
    //not all digits have been shifted the first time if `(x.len() % digits) == 0`.
    for start in 1..lowest_nonstart {
        let mut tmp = *x.get_unchecked(start);
        let mut i0 = start;
        let mut i1 = start + s;
        loop {
            tmp = std::mem::replace(x.get_unchecked_mut(i1), tmp);
            i0 = i1;
            if i1 >= wrap_point {
                i1 -= wrap_point;
                if i1 == start {
                    *x.get_unchecked_mut(i1) = tmp;
                    break
                }
            } else {
                i1 += s;
            }
        }
    }
}*/

/*
//Using backwards replacement with 2 indexes and 1 temporary used once per cycle
//A rotate left function that assumes `(1 <= s < x.len()) && (2 <= x.len() < 16)`
unsafe fn slice_rotate_right(x: &mut [u64], s: usize) {
    //in order to avoid allocation, we have to wrap around the indexing, and to directly copy between indexes we go backwards.
    let mut start = 0usize;
    let mut lowest_nonstart = s;
    loop {
        let end = start.wrapping_add(s);
        let tmp = *x.get_unchecked(start);
        let mut prev_i = start;
        let mut i = start.wrapping_add(x.len()).wrapping_sub(s);
        loop {
            *x.get_unchecked_mut(prev_i) = *x.get_unchecked(i);
            if i == end {
                *x.get_unchecked_mut(end) = tmp;
                break
            }
            prev_i = i;
            if i < s {
                //we cannot just have a `lowest_nonstart = i` here without a conditional `i < lowest_nonstart`, unless x.len() is below 16
                //(shortest counterexample is x.len() = 16 and s = 6)
                if i < lowest_nonstart {
                    lowest_nonstart = i;
                }
                i = i.wrapping_add(x.len()).wrapping_sub(s);
            } else {
                i = i.wrapping_sub(s);
            }
        }
        start = start.wrapping_add(1);
        if start == lowest_nonstart {return}
    }
    // not all elements have been shifted the first time.
    /*loop {
        let end = start + s;
        let tmp = x[start];
        let mut prev_i = start;
        let mut i = start + x.len() - s;
        loop {
            x[prev_i] = x[i];
            if i == end {
                x[end] = tmp;
                break
            }
            prev_i = i;
            if i < s {
                i += x.len() - s;
            } else {
                i -= s;
            }
        }
        start += 1;
        if start == lowest_nonstart {return}
    }*/
}
*/
/*
//A rotate left function that assumes `(1 <= s < x.len()) && (2 <= x.len() < 16)`
unsafe fn slice_rotate_right(x: &mut [u64], s: isize) {
    let x_len = x.len() as isize;
    let ptr: *mut u64 = x.as_mut_ptr();
    //in order to avoid allocation, we have to wrap around the indexing, and to directly copy between indexes we go backwards.
    let mut start = 0isize;
    let mut lowest_nonstart = s;
    loop {
        let end = start.wrapping_add(s);
        let mut prev_i = start;
        let mut i = start.wrapping_add(x_len).wrapping_sub(s);
        let tmp = std::ptr::read(ptr.offset(start));
        loop {
            std::ptr::copy_nonoverlapping(ptr.offset(i), ptr.offset(prev_i), 1);// *x.get_unchecked_mut(prev_i) = *x.get_unchecked(i);
            if i == end {
                std::ptr::write(ptr.offset(end), tmp);
                break
            }
            prev_i = i;
            if i < s {
                //we cannot just have a `lowest_nonstart = i` here without a conditional `i < lowest_nonstart`, unless x.len() is below 16
                //(shortest counterexample is x.len() = 16 and s = 6)
                if i < lowest_nonstart {
                    lowest_nonstart = i;
                }
                i = i.wrapping_add(x_len).wrapping_sub(s);
            } else {
                i = i.wrapping_sub(s);
            }
        }
        start = start.wrapping_add(1);
        if start == lowest_nonstart {return}
    }
    // not all elements have been shifted the first time.
    /*loop {
        let end = start + s;
        let tmp = x[start];
        let mut prev_i = start;
        let mut i = start + x.len() - s;
        loop {
            x[prev_i] = x[i];
            if i == end {
                x[end] = tmp;
                break
            }
            prev_i = i;
            if i < s {
                i += x.len() - s;
            } else {
                i -= s;
            }
        }
        start += 1;
        if start == lowest_nonstart {return}
    }*/
}*/

/*
//A rotate left function that assumes `(1 <= s < x.len()) && (2 <= x.len() < 16)`
unsafe fn slice_rotate_right(x: &mut [u64], s: usize) {
    let wrap_point = x.len() - s;
    //in order to avoid allocation, we have to "leapfrog" and wrap around the indexing.
    let mut lowest_nonstart = s;
    let mut tmp0 = *x.get_unchecked(0);
    let mut tmp1;
    let mut i = s;
    loop {
        tmp1 = *x.get_unchecked(i);
        *x.get_unchecked_mut(i) = tmp0;
        if i >= wrap_point {
            i -= wrap_point;
            if i == 0 {
                *x.get_unchecked_mut(i) = tmp1;
                break
            }
            //we cannot just have a `lowest_nonstart = i` here without a conditional `i < lowest_nonstart`, unless x.len() is below 16
            //(shortest counterexample is x.len() = 16 and s = 6)
            if i < lowest_nonstart {
                lowest_nonstart = i;
            }
        } else {
            i += s;
        }
        //Repeat with swapped temporaries in case the compiler cannot do it.
        //We do not need a second indexing variable to handle the source
        //digit, because that source variable is already stored in tmp0 or
        //tmp1.
        tmp0 = *x.get_unchecked(i);
        *x.get_unchecked_mut(i) = tmp1;
        if i >= wrap_point {
            i -= wrap_point;
            if i == 0 {
                *x.get_unchecked_mut(i) = tmp0;
                break
            }
            if i < lowest_nonstart {
                lowest_nonstart = i;
            }
        } else {
            i += s;
        }
    }
    //the leapfrog wraps around until it arrives back where it started, but
    //not all digits have been shifted the first time if `(x.len() % digits) == 0`.
    for start in 1..lowest_nonstart {
        let mut tmp0 = *x.get_unchecked(start);
        let mut tmp1;
        let mut i = start + s;
        loop {
            tmp1 = *x.get_unchecked(i);
            *x.get_unchecked_mut(i) = tmp0;
            if i >= wrap_point {
                i -= wrap_point;
                if i == start {
                    *x.get_unchecked_mut(i) = tmp1;
                    break
                }
            } else {
                i += s;
            }
            //Repeat with swapped temporaries in case the compiler cannot do it.
            //We do not need a second indexing variable to handle the source
            //digit, because that source variable is already stored in tmp0 or
            //tmp1.
            tmp0 = *x.get_unchecked(i);
            *x.get_unchecked_mut(i) = tmp1;
            if i >= wrap_point {
                i -= wrap_point;
                if i == start {
                    *x.get_unchecked_mut(i) = tmp0;
                    break
                }
            } else {
                i += s;
            }
        }
    }
}*/
/*
#[inline(never)]
fn gcd(lhs: usize, rhs: usize) -> usize {
    gcd_u16(lhs as u16, rhs as u16) as usize
}

fn gcd_u16(mut lhs: u16, mut rhs: u16) -> u16 {
    loop {
        if rhs == 0 {break lhs}
        lhs = lhs % rhs;
        if lhs == 0 {break rhs}
        rhs = rhs % lhs;
    }
}*/
/*
unsafe fn ptr_rotate<T>(mut left: usize, mid: *mut T, mut right: usize) {
    if mem::size_of::<T>() == 0 {
        return
    }
    if right == 0 {return}
    let mut x = mid.sub(left);
    //in order to avoid allocation, we have to "leapfrog" and wrap around the indexing.
    let mut tmp: T = x.read();
    let mut i = right;
    let mut lowest_nonstart = right;
    loop {
        tmp = x.add(i).replace(tmp);
        if i >= left {
            i -= left;
            if i == 0 {
                x.add(i).write(tmp);
                break
            }
            //we cannot just have a `lowest_nonstart = i` here without a conditional `i < lowest_nonstart`, unless x.len() is below 16
            //(shortest counterexample is x.len() = 16 and s = 6)
            if i < lowest_nonstart {
                lowest_nonstart = i;
            }
        } else {
            i += right;
        }
    }
    //the leapfrog wraps around until it arrives back where it started, but
    //not all digits have been shifted the first time if `(x.len() % digits) == 0`.
    for start in 1..lowest_nonstart {
        tmp = x.add(start).read();
        i = start + right;
        loop {
            tmp = x.add(i).replace(tmp);
            if i >= left {
                i -= left;
                if i == start {
                    x.add(i).write(tmp);
                    break
                }
            } else {
                i += right;
            }
        }
    }
}*/

/*
            let x = mid.sub(left);
            let mut start = x;
            let mut i1_start = mid;
            let mut end = x.add(right);
            let mut lowest_nonstart = end;
            let mut i0;
            let mut i1;
            let mut tmp: T;
            loop {
                i0 = start;
                i1 = i1_start;
                tmp = i0.read();
                loop {
                    //dbg!(left, right, lowest_nonstart, start);
                    i0.copy_from_nonoverlapping(i1, 1);
                    if i1 == end {
                        end.write(tmp);
                        break
                    }
                    if i1 < end {
                        if i1 < lowest_nonstart {
                            lowest_nonstart = i1;
                        }
                        i0 = i1.add(left);
                    } else {
                        i0 = i1.sub(right);
                    }
                    //dbg!(left, right, lowest_nonstart, start);
                    i1.copy_from_nonoverlapping(i0, 1);
                    if i0 == end {
                        end.write(tmp);
                        break
                    }
                    if i0 < end {
                        if i0 < lowest_nonstart {
                            lowest_nonstart = i0;
                        }
                        i1 = i0.add(left);
                    } else {
                        i1 = i0.sub(right);
                    }
                }
                start = start.add(1);
                i1_start = i1_start.add(1);
                if start == lowest_nonstart {return}
                end = end.add(1);
            }
*/

/*
test benches::x1024s128_new ... bench:         567 ns/iter (+/- 9)
test benches::x1024s128_old ... bench:         642 ns/iter (+/- 23)
test benches::x1024s256_new ... bench:         555 ns/iter (+/- 23)
test benches::x1024s256_old ... bench:         635 ns/iter (+/- 36)
test benches::x1024s32_new  ... bench:         610 ns/iter (+/- 29)
test benches::x1024s32_old  ... bench:         462 ns/iter (+/- 59)
test benches::x1024s512_new ... bench:         472 ns/iter (+/- 9)
test benches::x1024s512_old ... bench:         583 ns/iter (+/- 25)

        else {
            /*

if left >= right {

                loop {
                    ptr::swap_nonoverlapping(mid.sub(right), mid, right);
                    mid = mid.sub(right);
                    left -= right;
                    if left < right {
                        break
                    }
                }
            } else {
                loop {
                    ptr::swap_nonoverlapping(mid.sub(left), mid, left);
                    mid = mid.add(left);
                    right -= left;
                    if right < left {
                        break
                    }
                }
            }


            let (mut end, mut x, mut chunk, mut b) = if left >= right {
                let low = mid.sub(left).add(((left as u16) % (right as u16)) as usize);
                left -= right * (((left as u16) / (right as u16)) as usize);
                let mut x = low;
                (low, x, right, true)
            } else {
                let high = mid.add(right).sub(((right as u16) % (left as u16)) as usize);
                right -= left * (((right as u16) / (left as u16)) as usize);
                let mut x = high.sub(left);
                mid = mid.sub(left);
                (high, x, left, false)
            };
            loop {
                ptr::swap_nonoverlapping(x, mid, chunk);
                if b {
                    x = x.add(chunk);
                } else {
                    x = x.sub(chunk);
                }
                if x == mid {
                    mid = end;
                    break
                }
            }*/
            let delta = cmp::min(left, right);

            ptr::swap_nonoverlapping(
                mid.sub(left),
                mid.add(right - delta),
                delta);

            if left <= right {
                right -= delta;
            } else {
                left -= delta;
            }

            /*let chunk_size = gcd(left + right, right);
            let x = mid.sub(left);
            let mut y = x.add(right);
            loop {
                ptr::swap_nonoverlapping(x, y, chunk_size);
                if y == mid {
                    return
                }
                if y > mid {
                    y = y.sub(left);
                } else {
                    y = y.add(right);
                }
            }*/
            /*let chunk_size = gcd(left + right, right);
            let low = mid.sub(left).add(((left as u16) % (right as u16)) as usize);
            left -= right * (((left as u16) / (right as u16)) as usize);
            let y = mid.clone();
            let mut x = low;
            loop {
                ptr::swap_nonoverlapping(x, y, chunk_size);
                x = x.add(right);
                if x == y {
                    //mid = low;
                    return
                }
            }*/
            /*let chunk_size = right;
            let low = mid.sub(left).add(((left as u16) % (right as u16)) as usize);
            left -= right * (((left as u16) / (right as u16)) as usize);
            let mut x = low;
            loop {
                ptr::swap_nonoverlapping(x, mid, chunk_size);
                x = x.add(right);
                if x == mid {
                    mid = low;
                    break
                }
            }*/
        }*/

/*
test benches::x1024s128_new ... bench:         602 ns/iter (+/- 14)
test benches::x1024s128_old ... bench:         636 ns/iter (+/- 6)
test benches::x1024s256_new ... bench:         565 ns/iter (+/- 20)
test benches::x1024s256_old ... bench:         631 ns/iter (+/- 20)
test benches::x1024s32_new  ... bench:         612 ns/iter (+/- 8)
test benches::x1024s32_old  ... bench:         460 ns/iter (+/- 4)
test benches::x1024s512_new ... bench:         475 ns/iter (+/- 6)
test benches::x1024s512_old ... bench:         570 ns/iter (+/- 41)

test benches::x1024s128_new ... bench:         562 ns/iter (+/- 10)
test benches::x1024s128_old ... bench:         641 ns/iter (+/- 27)
test benches::x1024s256_new ... bench:         556 ns/iter (+/- 11)
test benches::x1024s256_old ... bench:         633 ns/iter (+/- 10)
test benches::x1024s32_new  ... bench:         608 ns/iter (+/- 7)
test benches::x1024s32_old  ... bench:         457 ns/iter (+/- 23)
test benches::x1024s512_new ... bench:         476 ns/iter (+/- 8)
test benches::x1024s512_old ... bench:         569 ns/iter (+/- 14)
*/

/*
if left >= right {
    let wrap_point = mid.sub(left).add(right);
    let x = mid.sub(left).add((left as u16) % (right as u16) as usize);
    loop {
        ptr::swap_nonoverlapping(mid.sub(right), mid, right);
        mid = mid.sub(right);
        left -= right;
        if mid < wrap_point {
            break
        }
    }
} else {
    let wrap_point = mid.add(right).sub(left);
    loop {
        swap(mid.sub(left), mid, left);
        ptr::swap_nonoverlapping(mid.sub(left), mid, left);
        mid = mid.add(left);
        right -= left;
        if mid > wrap_point {
            break
        }
    }
}
*/

// for some reason, all the 2 pointers, 1 temporary functions dramatically underperformed their swapping version
/*
// handle small rotations
let mut rawarray = MaybeUninit::<RawArray<T>>::uninit();
let buf = &mut (*rawarray.as_mut_ptr()).typed as *mut [T; 2] as *mut T;
let mut y = mid.sub(left);
ptr::copy_nonoverlapping(y, buf, chunk_size);
let mut x = mid;
loop {
    let wrap = mid.sub(left).add(right);
    ptr::copy_nonoverlapping(x, y, chunk_size);
    if y == mid.sub(left).add(right) {
        ptr::copy_nonoverlapping(buf, y, chunk_size);
        return
    }
    y = x;
    if x < wrap {
        x = x.add(left);
    } else {
        x = x.sub(right);
    }
}
*/
/*
/// Rotation is much faster if it has access to a little bit of memory. This
/// union provides a RawVec-like interface, but to a fixed-size stack buffer.
#[allow(unions_with_drop_fields)]
union RawArray<T> {
    /// Ensure this is appropriately aligned for T, and is big
    /// enough for two elements even if T is enormous.
    typed: [T; 2],
    /// For normally-sized types, especially things like u8, having more
    /// than 2 in the buffer is necessary for usefulness, so pad it out
    /// enough to be helpful, but not so big as to risk overflow.
    _extra: [usize; 32],
}

impl<T> RawArray<T> {
    fn cap() -> usize {
        if mem::size_of::<T>() == 0 {
            usize::max_value()
        } else {
            mem::size_of::<Self>() / mem::size_of::<T>()
        }
    }
}

unsafe fn ptr_rotate<T>(mut left: usize, mid: *mut T, mut right: usize) {
    if right == 0 {return}
    /*if left + right <= 16 {
        let mut x = mid.sub(left);
        let mut tmp: T = x.read();
        let mut i = right;
        // this can be found before hand by calculating gcd(left + right, right), but
        let mut lowest_nonstart = right;
        loop {
            tmp = x.add(i).replace(tmp);
            if i >= left {
                i -= left;
                if i == 0 {
                    x.add(i).write(tmp);
                    break
                }
                if i < lowest_nonstart {
                    lowest_nonstart = i;
                }
            } else {
                i += right;
            }
        }
        for start in 1..lowest_nonstart {
            tmp = x.add(start).read();
            i = start + right;
            loop {
                tmp = x.add(i).replace(tmp);
                if i >= left {
                    i -= left;
                    if i == start {
                        x.add(i).write(tmp);
                        break
                    }
                } else {
                    i += right;
                }
            }
        }
        return
    }*/

    loop {
        let delta = cmp::min(left, right);
        if delta <= RawArray::<T>::cap() {
            // We will always hit this immediately for ZST.
            break;
        }

        ptr::swap_nonoverlapping(
            mid.sub(left),
            mid.add(right - delta),
            delta);

        if left <= right {
            right -= delta;
        } else {
            left -= delta;
        }
    }

    let mut rawarray = MaybeUninit::<RawArray<T>>::uninit();
    let buf = &mut (*rawarray.as_mut_ptr()).typed as *mut [T; 2] as *mut T;
    let dim = mid.sub(left).add(right);
    if left <= right {
        ptr::copy_nonoverlapping(mid.sub(left), buf, left);
        ptr::copy(mid, mid.sub(left), right);
        ptr::copy_nonoverlapping(buf, dim, left);
    }
    else {
        ptr::copy_nonoverlapping(mid, buf, right);
        ptr::copy(mid.sub(left), dim, left);
        ptr::copy_nonoverlapping(buf, mid.sub(left), right);
    }
}*/

/*
#[inline(never)]
pub fn rotate_right(x: &mut [u64], s: usize) {
    if s >= x.len() {panic!("overshift")}
    if s == 0 {return}
    unsafe { slice_rotate_right(x,s); }
    return
    /*if x.len() < 32 {
        unsafe {slice_rotate_right(x,s); }
        return
    }
    let mut mid = x.len() - s;
    let mut lower = 0;
    let mut upper = x.len();
    unsafe {
    if ((x.len() - s) < (s / 4)) || (s < ((x.len() - s) / 4)) {slice_rotate_right(x, s); return}
    loop {
        let lower_diff = mid - lower;
        let upper_diff = upper - mid;
        if upper_diff < lower_diff {
            for i in mid..upper {
                let tmp = *x.get_unchecked(i);
                *x.get_unchecked_mut(i) = *x.get_unchecked(i - upper_diff);
                *x.get_unchecked_mut(i - upper_diff) = tmp;
            }
            upper = mid;
            mid = mid - upper_diff;
            if mid == lower {break}
        } else {
            //if upper_diff < (lower_diff / 4) {slice_rotate_right(&mut x[lower..upper],s); return}
            for i in lower..mid {
                let tmp = *x.get_unchecked(i);
                //if i + lower_diff >= x.len() {dbg!(lower_diff,i,lower,mid,upper);}
                *x.get_unchecked_mut(i) = *x.get_unchecked(i + lower_diff);
                *x.get_unchecked_mut(i + lower_diff) = tmp;
            }
            lower = mid;
            mid = mid + lower_diff;
            if mid == upper {break}
        }
    }
    }*/
}*/

