#pragma once

#ifndef _PDS_SSF_FV_SOA_H
#define _PDS_SSF_FV_SOA_H

#include "cuda_portability.h"
#include "Typedefs.h"

//------------------------------------------------------------------------------------------------------------
// SSF_FV_SoA_t class support
//
// This class implements Structure of Arrays (SoA) layout for better GPU memory coalescing
// compared to the original Array of Structures (AoS) layout in SSF_FV_t.
//
// Memory layout optimization:
// - All f1.x components are stored contiguously, followed by all f1.y, then f1.z
// - This allows warp threads to access consecutive memory addresses for better coalescing
//------------------------------------------------------------------------------------------------------------

class SSF_FV_SoA_t
{
   public:
      // Force components stored in SoA layout for better memory coalescing
      real8 *f1x, *f1y, *f1z;  ///< force on node 1 components <x,y,z>
      real8 *f2x, *f2y, *f2z;  ///< force on node 2 components <x,y,z>
      real8 *f3x, *f3y, *f3z;  ///< force on node 3 components <x,y,z>
      real8 *f4x, *f4y, *f4z;  ///< force on node 4 components <x,y,z>
      
      int np;                   ///< number of segment pairs (array size)

   public:
      __cuda_hdev__  SSF_FV_SoA_t(void);
      __cuda_hdev__  SSF_FV_SoA_t(int num_pairs);
      __cuda_hdev__ ~SSF_FV_SoA_t();

      // Memory management
      __cuda_hdev__ void Allocate(int num_pairs);
      __cuda_hdev__ void Deallocate(void);
      __cuda_hdev__ void Zero(void);
      
      // Conversion utilities
      __cuda_hdev__ void CopyFromAoS(const class SSF_FV_t *fv_aos, int num_pairs);
      __cuda_hdev__ void CopyToAoS(class SSF_FV_t *fv_aos, int num_pairs) const;
      
      // Access utilities for CUDA kernels
      __cuda_hdev__ void SetForce1(int i, real8 fx, real8 fy, real8 fz);
      __cuda_hdev__ void SetForce2(int i, real8 fx, real8 fy, real8 fz);
      __cuda_hdev__ void SetForce3(int i, real8 fx, real8 fy, real8 fz);
      __cuda_hdev__ void SetForce4(int i, real8 fx, real8 fy, real8 fz);
      
      __cuda_hdev__ void GetForce1(int i, real8 &fx, real8 &fy, real8 &fz) const;
      __cuda_hdev__ void GetForce2(int i, real8 &fx, real8 &fy, real8 &fz) const;
      __cuda_hdev__ void GetForce3(int i, real8 &fx, real8 &fy, real8 &fz) const;
      __cuda_hdev__ void GetForce4(int i, real8 &fx, real8 &fy, real8 &fz) const;
};

//------------------------------------------------------------------------------------------------------------
// Inline implementations
//------------------------------------------------------------------------------------------------------------

__cuda_hdev__  
inline 
SSF_FV_SoA_t::SSF_FV_SoA_t(void) 
{ 
   f1x = f1y = f1z = nullptr;
   f2x = f2y = f2z = nullptr;
   f3x = f3y = f3z = nullptr;
   f4x = f4y = f4z = nullptr;
   np = 0;
}

__cuda_hdev__  
inline 
SSF_FV_SoA_t::SSF_FV_SoA_t(int num_pairs) 
{ 
   f1x = f1y = f1z = nullptr;
   f2x = f2y = f2z = nullptr;
   f3x = f3y = f3z = nullptr;
   f4x = f4y = f4z = nullptr;
   np = 0;
   Allocate(num_pairs);
}

__cuda_hdev__  
inline 
SSF_FV_SoA_t::~SSF_FV_SoA_t() 
{
   Deallocate();
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::Allocate(int num_pairs)
{
   if (num_pairs <= 0) return;
   
   Deallocate(); // Clean up any existing allocation
   
   np = num_pairs;
   size_t size = np * sizeof(real8);
   
#ifdef __CUDA_ARCH__
   // Device allocation - this would need to be handled differently
   // For now, assume pointers are already allocated externally for device code
#else
   // Host allocation
   f1x = (real8*)malloc(size); f1y = (real8*)malloc(size); f1z = (real8*)malloc(size);
   f2x = (real8*)malloc(size); f2y = (real8*)malloc(size); f2z = (real8*)malloc(size);
   f3x = (real8*)malloc(size); f3y = (real8*)malloc(size); f3z = (real8*)malloc(size);
   f4x = (real8*)malloc(size); f4y = (real8*)malloc(size); f4z = (real8*)malloc(size);
#endif
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::Deallocate(void)
{
#ifdef __CUDA_ARCH__
   // Device deallocation - handled externally
#else
   // Host deallocation
   if (f1x) { free(f1x); f1x = nullptr; }
   if (f1y) { free(f1y); f1y = nullptr; }
   if (f1z) { free(f1z); f1z = nullptr; }
   if (f2x) { free(f2x); f2x = nullptr; }
   if (f2y) { free(f2y); f2y = nullptr; }
   if (f2z) { free(f2z); f2z = nullptr; }
   if (f3x) { free(f3x); f3x = nullptr; }
   if (f3y) { free(f3y); f3y = nullptr; }
   if (f3z) { free(f3z); f3z = nullptr; }
   if (f4x) { free(f4x); f4x = nullptr; }
   if (f4y) { free(f4y); f4y = nullptr; }
   if (f4z) { free(f4z); f4z = nullptr; }
#endif
   np = 0;
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::Zero(void)
{
   if (np <= 0) return;
   
   size_t size = np * sizeof(real8);
   memset(f1x, 0, size); memset(f1y, 0, size); memset(f1z, 0, size);
   memset(f2x, 0, size); memset(f2y, 0, size); memset(f2z, 0, size);
   memset(f3x, 0, size); memset(f3y, 0, size); memset(f3z, 0, size);
   memset(f4x, 0, size); memset(f4y, 0, size); memset(f4z, 0, size);
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::SetForce1(int i, real8 fx, real8 fy, real8 fz)
{
   if (i >= 0 && i < np) {
      f1x[i] = fx; f1y[i] = fy; f1z[i] = fz;
   }
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::SetForce2(int i, real8 fx, real8 fy, real8 fz)
{
   if (i >= 0 && i < np) {
      f2x[i] = fx; f2y[i] = fy; f2z[i] = fz;
   }
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::SetForce3(int i, real8 fx, real8 fy, real8 fz)
{
   if (i >= 0 && i < np) {
      f3x[i] = fx; f3y[i] = fy; f3z[i] = fz;
   }
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::SetForce4(int i, real8 fx, real8 fy, real8 fz)
{
   if (i >= 0 && i < np) {
      f4x[i] = fx; f4y[i] = fy; f4z[i] = fz;
   }
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::GetForce1(int i, real8 &fx, real8 &fy, real8 &fz) const
{
   if (i >= 0 && i < np) {
      fx = f1x[i]; fy = f1y[i]; fz = f1z[i];
   }
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::GetForce2(int i, real8 &fx, real8 &fy, real8 &fz) const
{
   if (i >= 0 && i < np) {
      fx = f2x[i]; fy = f2y[i]; fz = f2z[i];
   }
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::GetForce3(int i, real8 &fx, real8 &fy, real8 &fz) const
{
   if (i >= 0 && i < np) {
      fx = f3x[i]; fy = f3y[i]; fz = f3z[i];
   }
}

__cuda_hdev__ 
inline 
void SSF_FV_SoA_t::GetForce4(int i, real8 &fx, real8 &fy, real8 &fz) const
{
   if (i >= 0 && i < np) {
      fx = f4x[i]; fy = f4y[i]; fz = f4z[i];
   }
}

#endif  //  _PDS_SSF_FV_SOA_H

