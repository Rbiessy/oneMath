/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef _ONEMKL_SPARSE_BLAS_TYPES_HPP_
#define _ONEMKL_SPARSE_BLAS_TYPES_HPP_

#include "oneapi/mkl/types.hpp"
#include "detail/data_types.hpp"
#include "detail/operation_types.hpp"

/**
 * @file Include and define the sparse types that are common between close-source MKL API and oneMKL API.
*/

namespace oneapi {
namespace mkl {
namespace sparse {

enum class matrix_property : char {
    symmetric = 0x00,
    sorted = 0x01, /* CSR, CSC, BSR only */
};

enum class trsv_alg : char {
    default_alg = 0x00,
};

} // namespace sparse
} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_SPARSE_BLAS_TYPES_HPP_
