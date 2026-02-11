#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// use std::time::{Duration, Instant};
// use std::thread::sleep;

use macroquad::prelude::*;
use ocl::ProQue;


const WIDTH:  usize = 500;
const HEIGHT: usize = 350;
const TIME_STEPS: usize = 20;

const DIFFUSION_A: f64 = 1.0;
const DIFFUSION_B: f64 = 0.5;

const KF_SETTINGS: [(f64, f64); 4] = [
    (
        0.055,
        0.062
    ),
    (
        0.01923,
        0.04780
    ),
    (
        0.02725,
        0.06032
    ),
    (
        0.06483,
        0.06372
    )
];
const KF_DEFAULT: usize = 2;
const FEED: f64 = KF_SETTINGS[KF_DEFAULT].0;
const KILL: f64 = KF_SETTINGS[KF_DEFAULT].1;

// const COLOR_A: (f64, f64, f64) = (255.0, 255.0, 220.0);
// const COLOR_B: (f64, f64, f64) = (21.0, 78.0, 146.0);

const COLOR_A: (f64, f64, f64) = (33.0, 25.0, 81.0);
const COLOR_B: (f64, f64, f64) = (21.0, 245.0, 186.0);

// const COLOR_A: (f64, f64, f64) = (33.0, 25.0, 81.0);
// const COLOR_B: (f64, f64, f64) = (255.0, 245.0, 186.0);

// const COLOR_A: (f64, f64, f64) = (33.0, 32.0, 42.0);
// const COLOR_B: (f64, f64, f64) = (181.0, 70.0, 70.0);

const TEXT_COLOR: Color = Color::new(245.0 / 255.0, 0.0, 179.0 / 255.0, 1.0);

const UPSCALE: f64 = 2.0;
const WINDOW_WIDTH:  usize = (WIDTH as f64 * UPSCALE) as usize;
const WINDOW_HEIGHT: usize = (HEIGHT as f64 * UPSCALE) as usize;
// const TARGET_FPS: f32 = 165.0;

fn window_conf() -> Conf {
    return Conf {
        window_title: "Reaction Diffusion Algorithm".to_owned(),
        window_width: WINDOW_WIDTH as i32,
        window_height: WINDOW_HEIGHT as i32,
        window_resizable: false,
        high_dpi: false,
        platform: miniquad::conf::Platform {
            swap_interval: Some(0),
            ..Default::default()
        },
        ..Default::default()
    };
}

struct Canvas {
    pub data: Vec<f64>,
    rows: usize,
    columns: usize
}

impl Canvas {
    pub fn new(rows: usize, columns: usize) -> Self {
        let mut data = Vec::with_capacity(rows * columns * 2);
        for _ in 0..(rows * columns) {
            data.push(1.0);
            data.push(0.0);
        }

        return Self {
            data: data,
            rows: rows,
            columns: columns
        };
    }

    pub fn swap_with(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.data, &mut other.data);
    }

    fn get(&self, x: usize, y: usize, z: usize) -> f64 {
        return self.data[(x * self.columns + y) * 2 + z];
    }

    fn set(&mut self, x: usize, y: usize, z: usize, val: f64) {
        self.data[(x * self.columns + y) * 2 + z] = val;
    }

    pub fn laplace(&self, x: usize, y: usize, z: usize) -> f64 {
        let mut sum = 0.0;

        sum += self.get(x    , y    , z) * -1.0;
        sum += self.get(x - 1, y    , z) *  0.2;
        sum += self.get(x + 1, y    , z) *  0.2;
        sum += self.get(x    , y + 1, z) *  0.2;
        sum += self.get(x    , y - 1, z) *  0.2;
        sum += self.get(x - 1, y - 1, z) *  0.05;
        sum += self.get(x + 1, y - 1, z) *  0.05;
        sum += self.get(x + 1, y + 1, z) *  0.05;
        sum += self.get(x - 1, y + 1, z) *  0.05;

        return sum;
    }
}

const SRC: &str = "
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void func(
    __global double* grid,
    __global double* next,
    ulong WIDTH,
    ulong HEIGHT,
    double DA,
    double DB,
    double FEED,
    double KILL
) {
    int i = get_global_id(0);

    int y = i % HEIGHT;
    int x = (i - y) / HEIGHT;

    double a = grid[(x * HEIGHT + y) * 2];
    double b = grid[(x * HEIGHT + y) * 2 + 1];
    double ab2 = a * b * b;

    double la = 0.0;
    double lb = 0.0;

    la += grid[((x  ) * HEIGHT + (y  )) * 2    ] * -1.0;
    la += grid[((x-1) * HEIGHT + (y  )) * 2    ] *  0.2;
    la += grid[((x+1) * HEIGHT + (y  )) * 2    ] *  0.2;
    la += grid[((x  ) * HEIGHT + (y+1)) * 2    ] *  0.2;
    la += grid[((x  ) * HEIGHT + (y-1)) * 2    ] *  0.2;
    la += grid[((x-1) * HEIGHT + (y-1)) * 2    ] *  0.05;
    la += grid[((x+1) * HEIGHT + (y-1)) * 2    ] *  0.05;
    la += grid[((x+1) * HEIGHT + (y+1)) * 2    ] *  0.05;
    la += grid[((x-1) * HEIGHT + (y+1)) * 2    ] *  0.05;

    lb += grid[((x  ) * HEIGHT + (y  )) * 2 + 1] * -1.0;
    lb += grid[((x-1) * HEIGHT + (y  )) * 2 + 1] *  0.2;
    lb += grid[((x+1) * HEIGHT + (y  )) * 2 + 1] *  0.2;
    lb += grid[((x  ) * HEIGHT + (y+1)) * 2 + 1] *  0.2;
    lb += grid[((x  ) * HEIGHT + (y-1)) * 2 + 1] *  0.2;
    lb += grid[((x-1) * HEIGHT + (y-1)) * 2 + 1] *  0.05;
    lb += grid[((x+1) * HEIGHT + (y-1)) * 2 + 1] *  0.05;
    lb += grid[((x+1) * HEIGHT + (y+1)) * 2 + 1] *  0.05;
    lb += grid[((x-1) * HEIGHT + (y+1)) * 2 + 1] *  0.05;

    double nra =
          DA * la
        - ab2
        + FEED * (1.0 - a);
    double nrb =
          DB * lb
        + ab2
        - (KILL + FEED) * b;

// if x == 0 || x == WIDTH-1 || y == 0 || y == HEIGHT-1 {
    double valid = (double) !(x == 0 || x == WIDTH-1 || y == 0 || y == HEIGHT-1);

    next[(x * HEIGHT + y) * 2] = a + nra * valid;
    next[(x * HEIGHT + y) * 2 + 1] = b + nrb * valid;
}
";

#[macroquad::main(window_conf)]
async fn main() {
    // let frame_duration = Duration::from_secs_f32(1.0 / TARGET_FPS);

    // println!("{}", SRC);

    let pro_que = ProQue::builder()
        .src(SRC)
        .dims(WIDTH * HEIGHT)
        .build()
        .unwrap();

    let n = HEIGHT * WIDTH * 2;
    let buffer_grid = pro_que.buffer_builder::<f64>().len(n).build().unwrap();
    let buffer_next = pro_que.buffer_builder::<f64>().len(n).build().unwrap();

    let kernels: [ocl::Kernel; KF_SETTINGS.len()] = KF_SETTINGS
        .iter()
        .map(|(F, K)| {
            pro_que.kernel_builder("func")
                .arg(&buffer_grid)
                .arg(&buffer_next)
                .arg(WIDTH as u64)
                .arg(HEIGHT as u64)
                .arg(DIFFUSION_A)
                .arg(DIFFUSION_B)
                .arg(F)
                .arg(K)
                .build()
                .unwrap()
        })
        .collect::<Vec<ocl::Kernel>>()
        .try_into()
        .unwrap();

    let mut grid = Canvas::new(WIDTH, HEIGHT);
    let mut next = Canvas::new(WIDTH, HEIGHT);

    // let (cx, cy) = (WIDTH/2, HEIGHT/2);
    // for x in (cx-10)..(cx+10) {
    //     for y in (cy-2)..(cy+2) {
    //         grid.set(x, y, 1, 1.0);
    //         next.set(x, y, 1, 1.0);
    //     }
    // }

    buffer_grid.write(&grid.data).enq().unwrap();
    buffer_next.write(&next.data).enq().unwrap();

    let mut image = Image {
        bytes: vec![0u8; 4 * WIDTH * HEIGHT],
        width: WIDTH as u16,
        height: HEIGHT as u16
    };
    let texture = Texture2D::from_image(&image);
    texture.set_filter(FilterMode::Linear);

    let mut pause = false;
    let mut kf_mode = KF_DEFAULT;

    loop {
        // let frame_start = Instant::now();
        // let delta = get_frame_time() as f64 * TIME_SCALE;

        if is_mouse_button_down(MouseButton::Left) {
            let (mx, my) = mouse_position();
            let (gx, gy) = ((mx / WINDOW_WIDTH as f32 * WIDTH as f32) as isize, (my / WINDOW_HEIGHT as f32 * HEIGHT as f32) as isize);

            const R: isize = 3;
            for i in 0..(2 * R as isize + 1) {
                for j in 0..(2 * R as isize + 1) {
                    let (x, y) = (gx + i - R, gy + j - R);

                    if ((gx - x)).pow(2) + ((gy - y)).pow(2) <= R.pow(2) {
                        let (x, y) = (x.clamp(1, WIDTH as isize - 2) as usize, y.clamp(1, HEIGHT as isize - 2) as usize);

                        grid.set(x, y, 0, 0.0);
                        grid.set(x, y, 1, 1.0);
                    }
                }
            }
        }
        else if is_mouse_button_down(MouseButton::Right) {
            let (mx, my) = mouse_position();
            let (gx, gy) = ((mx / WINDOW_WIDTH as f32 * WIDTH as f32) as isize, (my / WINDOW_HEIGHT as f32 * HEIGHT as f32) as isize);

            const R: isize = 7;
            for i in 0..(2 * R as isize + 1) {
                for j in 0..(2 * R as isize + 1) {
                    let (x, y) = ((gx + i - R).max(0).min(WIDTH as isize - 1), (gy + j - R).max(0).min(HEIGHT as isize - 1));

                    if (gx - x).pow(2) + (gy - y).pow(2) <= R.pow(2) {
                        let (x, y) = (x as usize, y as usize);

                        grid.set(x, y, 0, 1.0);
                        grid.set(x, y, 1, 0.0);
                    }
                }
            }
        }

        if get_keys_pressed().contains(&macroquad::input::KeyCode::Space) {
            pause = !pause;
        }

        if is_mouse_button_pressed(MouseButton::Middle) {
            kf_mode = (kf_mode + 1) % KF_SETTINGS.len();
        }

        for _ in 0..(TIME_STEPS/*-1*/) {
            if pause {
                continue;
            }

            buffer_grid.write(&grid.data).enq().unwrap();

            unsafe { kernels[kf_mode].enq().unwrap(); }

            buffer_next.read(&mut grid.data).enq().unwrap();

            // for y in 1..(HEIGHT-1) {
            //     for x in 1..(WIDTH-1) {
            //         let a = grid.get(x, y, 0);
            //         let b = grid.get(x, y, 1);
            //
            //         let next_a = a
            //             + (DIFFUSION_A * grid.laplace(x, y, 0))
            //             - (a * b * b)
            //             + (FEED * (1.0 - a));
            //         let next_b = b
            //             + (DIFFUSION_B * grid.laplace(x, y, 1))
            //             + (a * b * b)
            //             - ((KILL + FEED) * b);
            //
            //         next.set(x, y, 0, next_a);
            //         next.set(x, y, 1, next_b);
            //     }
            // }
            //
            // grid.swap_with(&mut next);
        }

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let a = grid.get(x, y, 0);
                let b = grid.get(x, y, 1);

                let index = (WIDTH * y + x) * 4;

                let c = (a - b).clamp(0.0, 1.0);
                let r_ = (COLOR_B.0 + (COLOR_A.0 - COLOR_B.0) * c) as u8;
                let g_ = (COLOR_B.1 + (COLOR_A.1 - COLOR_B.1) * c) as u8;
                let b_ = (COLOR_B.2 + (COLOR_A.2 - COLOR_B.2) * c) as u8;

                // let l = (c * 255.0) as u8;
                image.bytes[index..index+4].copy_from_slice(&[r_, g_, b_, 255]);

                // if x == 0 || x == WIDTH-1 || y == 0 || y == HEIGHT-1 {
                //     continue;
                // }
                //
                // let next_a = a + /*delta **/ (
                //       (DIFFUSION_A * grid.laplace(x, y, 0))
                //     - (a * b * b)
                //     + (FEED * (1.0 - a)));
                // let next_b = b + /*delta **/ (
                //       (DIFFUSION_B * grid.laplace(x, y, 1))
                //     + (a * b * b)
                //     - ((KILL + FEED) * b));
                //
                // next.set(x, y, 0, next_a);
                // next.set(x, y, 1, next_b);
            }
        }

        // grid.swap_with(&mut next);

        texture.update(&image);

        draw_texture_ex(
            &texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32)),
                ..Default::default()
            },
        );

        draw_text(&format!("{:.2}", get_fps()), 5.0, 25.0, 35.0, TEXT_COLOR);

        next_frame().await;

        // let elapsed = frame_start.elapsed();
        // if elapsed < frame_duration {
        //     sleep(frame_duration - elapsed);
        // }
    }
}

