// Include the boost unit testing framework,
// requires -lboost_unit_test_framework when linking
#include <boost/test/included/unit_test.hpp>

#include "sphere.h"

// define the name of the test suite
BOOST_AUTO_TEST_SUITE (spheretest)

BOOST_AUTO_TEST_CASE (inittest)
{
    DEM testDEM();
}

BOOST_AUTO_TEST_SUITE_END()


