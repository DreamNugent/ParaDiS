#include "SSF_FV_SoA_t.h"
#include "SSF_FV_t.h"
#include <cstring>

//------------------------------------------------------------------------------------------------------------
// Conversion utilities implementation
//------------------------------------------------------------------------------------------------------------

__cuda_hdev__ 
void SSF_FV_SoA_t::CopyFromAoS(const SSF_FV_t *fv_aos, int num_pairs)
{
   if (!fv_aos || num_pairs <= 0) return;
   
   // Ensure we have the right size allocated
   if (np != num_pairs) {
      Allocate(num_pairs);
   }
   
   // Convert from AoS to SoA layout
   for (int i = 0; i < num_pairs; i++) {
      f1x[i] = fv_aos[i].f1[0]; f1y[i] = fv_aos[i].f1[1]; f1z[i] = fv_aos[i].f1[2];
      f2x[i] = fv_aos[i].f2[0]; f2y[i] = fv_aos[i].f2[1]; f2z[i] = fv_aos[i].f2[2];
      f3x[i] = fv_aos[i].f3[0]; f3y[i] = fv_aos[i].f3[1]; f3z[i] = fv_aos[i].f3[2];
      f4x[i] = fv_aos[i].f4[0]; f4y[i] = fv_aos[i].f4[1]; f4z[i] = fv_aos[i].f4[2];
   }
}

__cuda_hdev__ 
void SSF_FV_SoA_t::CopyToAoS(SSF_FV_t *fv_aos, int num_pairs) const
{
   if (!fv_aos || num_pairs <= 0 || num_pairs > np) return;
   
   // Convert from SoA to AoS layout
   for (int i = 0; i < num_pairs; i++) {
      fv_aos[i].f1[0] = f1x[i]; fv_aos[i].f1[1] = f1y[i]; fv_aos[i].f1[2] = f1z[i];
      fv_aos[i].f2[0] = f2x[i]; fv_aos[i].f2[1] = f2y[i]; fv_aos[i].f2[2] = f2z[i];
      fv_aos[i].f3[0] = f3x[i]; fv_aos[i].f3[1] = f3y[i]; fv_aos[i].f3[2] = f3z[i];
      fv_aos[i].f4[0] = f4x[i]; fv_aos[i].f4[1] = f4y[i]; fv_aos[i].f4[2] = f4z[i];
   }
}

