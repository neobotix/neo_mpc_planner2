#ifndef INCLUDE_COSTMAPWRAPPER_H_
#define INCLUDE_COSTMAPWRAPPER_H_

#include "NeoMpcPlanner.h"
#include <pybind11/pybind11.h>
#include <string>

class CostmapWrapper
{
public:
  CostmapWrapper() = default;
  std::string shareCostMap();
};

#endif /* INCLUDE_NEOLOCALPLANNER_H_ */
