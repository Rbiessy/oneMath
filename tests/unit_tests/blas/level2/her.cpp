/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "cblas.h"
#include "oneapi/math.hpp"
#include "oneapi/math/detail/config.hpp"
#include "onemath_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

extern std::vector<sycl::device*> devices;

namespace {

template <typename fp, typename fp_scalar>
int test(device* dev, oneapi::math::layout layout, oneapi::math::uplo upper_lower, int n,
         fp_scalar alpha, int incx, int lda) {
    // Prepare data.
    vector<fp> x, A_ref, A;
    rand_vector(x, n, incx);
    rand_matrix(A, layout, oneapi::math::transpose::nontrans, n, n, lda);
    A_ref = A;

    // Call Reference HER.
    const int n_ref = n, incx_ref = incx, lda_ref = lda;
    using fp_ref = typename ref_type_info<fp>::type;
    using fp_scalar_ref = typename ref_type_info<fp_scalar>::type;

    ::her(convert_to_cblas_layout(layout), convert_to_cblas_uplo(upper_lower), &n_ref,
          (fp_scalar_ref*)&alpha, (fp_ref*)x.data(), &incx_ref, (fp_ref*)A_ref.data(), &lda_ref);

    // Call DPC++ HER.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during HER:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp, 1> A_buffer = make_buffer(A);

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::her(main_queue, upper_lower, n, alpha, x_buffer,
                                                      incx, A_buffer, lda);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::her(main_queue, upper_lower, n, alpha, x_buffer,
                                                   incx, A_buffer, lda);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::her,
                                        upper_lower, n, alpha, x_buffer, incx, A_buffer, lda);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::her, upper_lower,
                                        n, alpha, x_buffer, incx, A_buffer, lda);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during HER:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of HER:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto A_accessor = A_buffer.get_host_access(read_only);
    bool good = check_equal_matrix(A_accessor, A_ref, layout, n, n, lda, n, std::cout);

    return (int)good;
}

class HerTests : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {
};

TEST_P(HerTests, ComplexSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                          oneapi::math::uplo::lower, 30, alpha, 2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                          oneapi::math::uplo::upper, 30, alpha, 2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                          oneapi::math::uplo::lower, 30, alpha, -2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                          oneapi::math::uplo::upper, 30, alpha, -2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                          oneapi::math::uplo::lower, 30, alpha, 1, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                          oneapi::math::uplo::upper, 30, alpha, 1, 42)));
}
TEST_P(HerTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    double alpha(2.0);
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                            oneapi::math::uplo::lower, 30, alpha, 2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                            oneapi::math::uplo::upper, 30, alpha, 2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                            oneapi::math::uplo::lower, 30, alpha, -2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                            oneapi::math::uplo::upper, 30, alpha, -2, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                            oneapi::math::uplo::lower, 30, alpha, 1, 42)));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                            oneapi::math::uplo::upper, 30, alpha, 1, 42)));
}

INSTANTIATE_TEST_SUITE_P(HerTestSuite, HerTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
