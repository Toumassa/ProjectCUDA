// =========================================================================================
//    Structured Class-Label in Random Forests. This is a re-implementation of
//    the work we presented at ICCV'11 in Barcelona, Spain.
//
//    In case of using this code, please cite the following paper:
//    P. Kontschieder, S. Rota Bulò, H. Bischof and M. Pelillo.
//    Structured Class-Labels in Random Forests for Semantic Image Labelling. In (ICCV), 2011.
//
//    Implementation by Peter Kontschieder and Samuel Rota Bulò
//    October 2013
//
// =========================================================================================

#ifndef GLOBAL_H_
#define GLOBAL_H_


#define DEBUGX	494
#define DEBUGY	342

#ifdef WIN32
  //here only windows
  #include <stdint.h>
  #include <direct.h>
  #include <io.h>
#else
  // here not windows
  #define _copysign copysign
  #define _mkdir(X) mkdir(X,S_IRWXU|S_IRGRP|S_IXGRP)
  #define _access access
  #define _finite finite
  #define sprintf_s snprintf
#endif

#endif /* GLOBAL_H_ */
