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


int AdjointVTI(struct dataobj *restrict damp_vec, struct dataobj *restrict delta_vec, const float dt, struct dataobj *restrict epsilon_vec, const float o_x, const float o_y, struct dataobj *restrict p_vec, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict srca_vec, struct dataobj *restrict srca_coords_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int x_size, const int y_M, const int y_m, const int y_size, const int p_rec_M, const int p_rec_m, const int p_srca_M, const int p_srca_m, const int time_M, const int time_m, struct profiler * timers)
{
  float *r6_vec;
  posix_memalign((void**)(&r6_vec),64,(x_size + 4)*(y_size + 4)*sizeof(float));
  float *r7_vec;
  posix_memalign((void**)(&r7_vec),64,(x_size + 4)*(y_size + 4)*sizeof(float));

  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*restrict delta)[delta_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[delta_vec->size[1]]) delta_vec->data;
  float (*restrict epsilon)[epsilon_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[epsilon_vec->size[1]]) epsilon_vec->data;
  float (*restrict p)[p_vec->size[1]][p_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[p_vec->size[1]][p_vec->size[2]]) p_vec->data;
  float (*restrict r6)[y_size + 4] __attribute__ ((aligned (64))) = (float (*)[y_size + 4]) r6_vec;
  float (*restrict r7)[y_size + 4] __attribute__ ((aligned (64))) = (float (*)[y_size + 4]) r7_vec;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict srca)[srca_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[srca_vec->size[1]]) srca_vec->data;
  float (*restrict srca_coords)[srca_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[srca_coords_vec->size[1]]) srca_coords_vec->data;
  float (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/(dt*dt);
  float r1 = 1.0F/dt;

  for (int time = time_M, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time >= time_m; time -= 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    /* Begin section0 */
    START_TIMER(section0)
    for (int x = x_m - 2; x <= x_M + 2; x += 1)
    {
      #pragma omp simd aligned(delta,epsilon,p:32)
      for (int y = y_m - 2; y <= y_M + 2; y += 1)
      {
        float r14 = 2.0F*epsilon[x + 4][y + 4];
        float r13 = 1.25e-1F*p[t0][x + 4][y + 2] - p[t0][x + 4][y + 3] + p[t0][x + 4][y + 5] - 1.25e-1F*p[t0][x + 4][y + 6];
        float r12 = 1.25e-1F*p[t0][x + 2][y + 4] - p[t0][x + 3][y + 4] + p[t0][x + 5][y + 4] - 1.25e-1F*p[t0][x + 6][y + 4];
        float r11 = 6.25000009e-3F*p[t0][x + 4][y + 2] - 5.0e-2F*p[t0][x + 4][y + 3] + 5.0e-2F*p[t0][x + 4][y + 5] - 6.25000009e-3F*p[t0][x + 4][y + 6];
        float r10 = 6.25000009e-3F*p[t0][x + 2][y + 4] - 5.0e-2F*p[t0][x + 3][y + 4] + 5.0e-2F*p[t0][x + 5][y + 4] - 6.25000009e-3F*p[t0][x + 6][y + 4];
        float r9 = 4.16666673e-3F*p[t0][x + 4][y + 2] - 3.33333338e-2F*p[t0][x + 4][y + 3] + 3.33333338e-2F*p[t0][x + 4][y + 5] - 4.16666673e-3F*p[t0][x + 4][y + 6];
        float r8 = (2.46913615111777e-6F*(r12*r12)*(r13*r13)*(delta[x + 4][y + 4] - epsilon[x + 4][y + 4]))/(1.97530864e-1F*(r10*r10)*(r11*r11)*(r14 + 2.0F) + 1.97530864e-1F*(r10*r10*r10*r10)*(2*epsilon[x + 4][y + 4] + 1.0F) + r9*r9*r9*r9 + 1.19209e-7F);
        r6[x + 2][y + 2] = (r8 + 1.0F)*p[t0][x + 4][y + 4];
        r7[x + 2][y + 2] = (r14 + r8 + 1.0F)*p[t0][x + 4][y + 4];
      }
    }
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(damp,p,vp:32)
      for (int y = y_m; y <= y_M; y += 1)
      {
        float r15 = 1.0F/(vp[x + 4][y + 4]*vp[x + 4][y + 4]);
        p[t1][x + 4][y + 4] = (r1*damp[x + 1][y + 1]*p[t0][x + 4][y + 4] + r15*(-r0*(-2.0F*p[t0][x + 4][y + 4]) - r0*p[t2][x + 4][y + 4]) - 6.24999986e-3F*(r6[x + 2][y + 2] + r7[x + 2][y + 2]) + 2.08333329e-4F*(-r6[x + 2][y] - r6[x + 2][y + 4] - r7[x][y + 2] - r7[x + 4][y + 2]) + 3.33333326e-3F*(r6[x + 2][y + 1] + r6[x + 2][y + 3] + r7[x + 1][y + 2] + r7[x + 3][y + 2]))/(r0*r15 + r1*damp[x + 1][y + 1]);
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */

    /* Begin section1 */
    START_TIMER(section1)
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
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1)
      {
        float r2 = (dt*dt)*(vp[ii_rec_0 + 4][ii_rec_1 + 4]*vp[ii_rec_0 + 4][ii_rec_1 + 4])*(2.5e-3F*px*py - 5.0e-2F*px - 5.0e-2F*py + 1)*rec[time][p_rec];
        p[t1][ii_rec_0 + 4][ii_rec_1 + 4] += r2;
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= y_M + 1)
      {
        float r3 = (dt*dt)*(vp[ii_rec_0 + 4][ii_rec_2 + 4]*vp[ii_rec_0 + 4][ii_rec_2 + 4])*(-2.5e-3F*px*py + 5.0e-2F*py)*rec[time][p_rec];
        p[t1][ii_rec_0 + 4][ii_rec_2 + 4] += r3;
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        float r4 = (dt*dt)*(vp[ii_rec_3 + 4][ii_rec_1 + 4]*vp[ii_rec_3 + 4][ii_rec_1 + 4])*(-2.5e-3F*px*py + 5.0e-2F*px)*rec[time][p_rec];
        p[t1][ii_rec_3 + 4][ii_rec_1 + 4] += r4;
      }
      if (ii_rec_2 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_2 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        float r5 = 2.5e-3F*px*py*(dt*dt)*(vp[ii_rec_3 + 4][ii_rec_2 + 4]*vp[ii_rec_3 + 4][ii_rec_2 + 4])*rec[time][p_rec];
        p[t1][ii_rec_3 + 4][ii_rec_2 + 4] += r5;
      }
    }
    STOP_TIMER(section1,timers)
    /* End section1 */

    /* Begin section2 */
    START_TIMER(section2)
    for (int p_srca = p_srca_m; p_srca <= p_srca_M; p_srca += 1)
    {
      float posx = -o_x + srca_coords[p_srca][0];
      float posy = -o_y + srca_coords[p_srca][1];
      int ii_srca_0 = (int)(floor(5.0e-2F*posx));
      int ii_srca_1 = (int)(floor(5.0e-2F*posy));
      int ii_srca_2 = 1 + (int)(floor(5.0e-2F*posy));
      int ii_srca_3 = 1 + (int)(floor(5.0e-2F*posx));
      float px = (float)(posx - 2.0e+1F*(int)(floor(5.0e-2F*posx)));
      float py = (float)(posy - 2.0e+1F*(int)(floor(5.0e-2F*posy)));
      float sum = 0.0F;
      if (ii_srca_0 >= x_m - 1 && ii_srca_1 >= y_m - 1 && ii_srca_0 <= x_M + 1 && ii_srca_1 <= y_M + 1)
      {
        sum += (2.5e-3F*px*py - 5.0e-2F*px - 5.0e-2F*py + 1)*p[t0][ii_srca_0 + 4][ii_srca_1 + 4];
      }
      if (ii_srca_0 >= x_m - 1 && ii_srca_2 >= y_m - 1 && ii_srca_0 <= x_M + 1 && ii_srca_2 <= y_M + 1)
      {
        sum += (-2.5e-3F*px*py + 5.0e-2F*py)*p[t0][ii_srca_0 + 4][ii_srca_2 + 4];
      }
      if (ii_srca_1 >= y_m - 1 && ii_srca_3 >= x_m - 1 && ii_srca_1 <= y_M + 1 && ii_srca_3 <= x_M + 1)
      {
        sum += (-2.5e-3F*px*py + 5.0e-2F*px)*p[t0][ii_srca_3 + 4][ii_srca_1 + 4];
      }
      if (ii_srca_2 >= y_m - 1 && ii_srca_3 >= x_m - 1 && ii_srca_2 <= y_M + 1 && ii_srca_3 <= x_M + 1)
      {
        sum += 2.5e-3F*px*py*p[t0][ii_srca_3 + 4][ii_srca_2 + 4];
      }
      srca[time][p_srca] = sum;
    }
    STOP_TIMER(section2,timers)
    /* End section2 */
  }

  free(r6_vec);
  free(r7_vec);

  return 0;
}
/* Backdoor edit at Fri Sep 23 03:10:45 2022*/ 
