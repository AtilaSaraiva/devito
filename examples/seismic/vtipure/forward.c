#define _POSIX_C_SOURCE 200809L
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct dataobj
{
  void *restrict data;
  int * size;
  int * npsize;
  int * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
} ;


int ForwardVTI(struct dataobj *restrict damp_vec, struct dataobj *restrict delta_vec, const float dt, struct dataobj *restrict epsilon_vec, const float o_x, const float o_y, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*restrict delta)[delta_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[delta_vec->size[1]]) delta_vec->data;
  float (*restrict epsilon)[epsilon_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[epsilon_vec->size[1]]) epsilon_vec->data;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/(dt*dt);
  float r1 = 1.0F/dt;

  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    /* Begin section0 */
    START_TIMER(section0)
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(damp,delta,epsilon,u,vp:32)
      for (int y = y_m; y <= y_M; y += 1)
      {
        float r14 = -6.24999986e-3F*u[t0][x + 4][y + 4];
        float r13 = 2.0F*epsilon[x + 4][y + 4];
        float r12 = 1.25e-1F*u[t0][x + 4][y + 2] - u[t0][x + 4][y + 3] + u[t0][x + 4][y + 5] - 1.25e-1F*u[t0][x + 4][y + 6];
        float r11 = 1.25e-1F*u[t0][x + 2][y + 4] - u[t0][x + 3][y + 4] + u[t0][x + 5][y + 4] - 1.25e-1F*u[t0][x + 6][y + 4];
        float r10 = 1.0F/(vp[x + 4][y + 4]*vp[x + 4][y + 4]);
        float r9 = 6.25000009e-3F*u[t0][x + 4][y + 2] - 5.0e-2F*u[t0][x + 4][y + 3] + 5.0e-2F*u[t0][x + 4][y + 5] - 6.25000009e-3F*u[t0][x + 4][y + 6];
        float r8 = 6.25000009e-3F*u[t0][x + 2][y + 4] - 5.0e-2F*u[t0][x + 3][y + 4] + 5.0e-2F*u[t0][x + 5][y + 4] - 6.25000009e-3F*u[t0][x + 6][y + 4];
        float r7 = 4.16666673e-3F*u[t0][x + 4][y + 2] - 3.33333338e-2F*u[t0][x + 4][y + 3] + 3.33333338e-2F*u[t0][x + 4][y + 5] - 4.16666673e-3F*u[t0][x + 4][y + 6];
        float r6 = (2.46913615111777e-6F*(r11*r11)*(r12*r12)*(delta[x + 4][y + 4] - epsilon[x + 4][y + 4]))/(r7*r7*r7*r7 + 1.97530864e-1F*(r8*r8)*(r9*r9)*(r13 + 2.0F) + 1.97530864e-1F*(r8*r8*r8*r8)*(2*epsilon[x + 4][y + 4] + 1.0F) + 1.19209e-7F);
        u[t2][x + 4][y + 4] = (r1*damp[x + 1][y + 1]*u[t0][x + 4][y + 4] + r10*(-r0*(-2.0F*u[t0][x + 4][y + 4]) - r0*u[t1][x + 4][y + 4]) + (r6 + 1.0F)*(r14 - 2.08333329e-4F*(u[t0][x + 4][y + 2] + u[t0][x + 4][y + 6]) + 3.33333326e-3F*(u[t0][x + 4][y + 3] + u[t0][x + 4][y + 5])) + (r13 + r6 + 1.0F)*(r14 - 2.08333329e-4F*(u[t0][x + 2][y + 4] + u[t0][x + 6][y + 4]) + 3.33333326e-3F*(u[t0][x + 3][y + 4] + u[t0][x + 5][y + 4])))/(r0*r10 + r1*damp[x + 1][y + 1]);
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */

    /* Begin section1 */
    START_TIMER(section1)
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      float posx = -o_x + src_coords[p_src][0];
      float posy = -o_y + src_coords[p_src][1];
      int ii_src_0 = (int)(floor(5.0e-2F*posx));
      int ii_src_1 = (int)(floor(5.0e-2F*posy));
      int ii_src_2 = 1 + (int)(floor(5.0e-2F*posy));
      int ii_src_3 = 1 + (int)(floor(5.0e-2F*posx));
      float px = (float)(posx - 2.0e+1F*(int)(floor(5.0e-2F*posx)));
      float py = (float)(posy - 2.0e+1F*(int)(floor(5.0e-2F*posy)));
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1)
      {
        float r2 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_1 + 4]*vp[ii_src_0 + 4][ii_src_1 + 4])*(2.5e-3F*px*py - 5.0e-2F*px - 5.0e-2F*py + 1)*src[time][p_src];
        u[t2][ii_src_0 + 4][ii_src_1 + 4] += r2;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= y_M + 1)
      {
        float r3 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_2 + 4]*vp[ii_src_0 + 4][ii_src_2 + 4])*(-2.5e-3F*px*py + 5.0e-2F*py)*src[time][p_src];
        u[t2][ii_src_0 + 4][ii_src_2 + 4] += r3;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        float r4 = (dt*dt)*(vp[ii_src_3 + 4][ii_src_1 + 4]*vp[ii_src_3 + 4][ii_src_1 + 4])*(-2.5e-3F*px*py + 5.0e-2F*px)*src[time][p_src];
        u[t2][ii_src_3 + 4][ii_src_1 + 4] += r4;
      }
      if (ii_src_2 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_2 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        float r5 = 2.5e-3F*px*py*(dt*dt)*(vp[ii_src_3 + 4][ii_src_2 + 4]*vp[ii_src_3 + 4][ii_src_2 + 4])*src[time][p_src];
        u[t2][ii_src_3 + 4][ii_src_2 + 4] += r5;
      }
    }
    STOP_TIMER(section1,timers)
    /* End section1 */

    /* Begin section2 */
    START_TIMER(section2)
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
    {
      float posx = -o_x + rec_coords[p_rec][0];
      float posy = -o_y + rec_coords[p_rec][1];
      int ii_rec_0 = (int)(floor(5.0e-2F*posx));
      int ii_rec_1 = (int)(floor(5.0e-2F*posy));
      int ii_rec_2 = 1 + (int)(floor(5.0e-2F*posy));
      int ii_rec_3 = 1 + (int)(floor(5.0e-2F*posx));
      float px = (float)(posx - 2.0e+1F*(int)(floor(5.0e-2F*posx)));
      float py = (float)(posy - 2.0e+1F*(int)(floor(5.0e-2F*posy)));
      float sum = 0.0F;
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1)
      {
        sum += (2.5e-3F*px*py - 5.0e-2F*px - 5.0e-2F*py + 1)*u[t0][ii_rec_0 + 4][ii_rec_1 + 4];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= y_M + 1)
      {
        sum += (-2.5e-3F*px*py + 5.0e-2F*py)*u[t0][ii_rec_0 + 4][ii_rec_2 + 4];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        sum += (-2.5e-3F*px*py + 5.0e-2F*px)*u[t0][ii_rec_3 + 4][ii_rec_1 + 4];
      }
      if (ii_rec_2 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_2 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        sum += 2.5e-3F*px*py*u[t0][ii_rec_3 + 4][ii_rec_2 + 4];
      }
      rec[time][p_rec] = sum;
    }
    STOP_TIMER(section2,timers)
    /* End section2 */
  }

  return 0;
}
/* Backdoor edit at Fri Sep 23 03:10:41 2022*/ 
