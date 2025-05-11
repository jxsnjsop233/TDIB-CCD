# pragma once
#include "utils.h"
template<typename ParamObj1, typename ParamObj2, typename ParamBound1, typename ParamBound2>
class SolverTrad{
	static bool primitiveCheck(const ParamObj1 &CpPos1, const ParamObj1 &CpVel1, 
							const ParamObj2 &CpPos2, const ParamObj2 &CpVel2,
							const ParamBound1 &divUvB1, const ParamBound2 &divUvB2,
							const BoundingBoxType& bb,
							const Array2d divTime = Array2d(0,DeltaT)) {
		auto posStart1 = CpPos1.divideBezierPatch(divUvB1), posEnd1 = posStart1;
		auto ptVel1 = CpVel1.divideBezierPatch(divUvB1);
		auto posStart2 = CpPos2.divideBezierPatch(divUvB2), posEnd2 = posStart2;
		auto ptVel2 = CpVel2.divideBezierPatch(divUvB2);
		for(int i=0;i<ParamObj1::cntCp;i++){
			posStart1[i]+=ptVel1[i]*divTime[0],
			posEnd1[i]+=ptVel1[i]*divTime[1];
		}
		for(int i=0;i<ParamObj2::cntCp;i++){
			posStart2[i]+=ptVel2[i]*divTime[0],
			posEnd2[i]+=ptVel2[i]*divTime[1];
		}

		std::vector<Vector3d> axes;
		axes.clear();
		setAxes<ParamObj1, ParamObj2>(posStart1, posStart2, axes, bb);

		for(auto& axis:axes){
			double maxProj1 = -std::numeric_limits<double>::infinity(), minProj1 = std::numeric_limits<double>::infinity();
			for(const auto&p:posStart1){
				maxProj1 = std::max(maxProj1, p.dot(axis));
				minProj1 = std::min(minProj1, p.dot(axis));
			}
			for(const auto&p:posEnd1){
				maxProj1 = std::max(maxProj1, p.dot(axis));
				minProj1 = std::min(minProj1, p.dot(axis));
			}
			double maxProj2 = -std::numeric_limits<double>::infinity(), minProj2 = std::numeric_limits<double>::infinity();
			for(const auto&p:posStart2){
				maxProj2 = std::max(maxProj2, p.dot(axis));
				minProj2 = std::min(minProj2, p.dot(axis));
			}
			for(const auto&p:posEnd2){
				maxProj2 = std::max(maxProj2, p.dot(axis));
				minProj2 = std::min(minProj2, p.dot(axis));
			}
			if(maxProj2<minProj1 || maxProj1<minProj2) return false;
		}
		return true;
	}
public:
		static double solveCCD(const ParamObj1 &CpPos1, const ParamObj1 &CpVel1, 
						const ParamObj2 &CpPos2, const ParamObj2 &CpVel2,
						Array2d& uv1, Array2d& uv2, 
						const BoundingBoxType& bb,
						const double deltaDist,
						const double upperTime = DeltaT) {
		struct PatchPair{
			ParamBound1 pb1;
			ParamBound2 pb2;
			Array2d tIntv;
			PatchPair(const ParamBound1& c1, const ParamBound2& c2, 
					Array2d t = Array2d(0,DeltaT)): pb1(c1), pb2(c2), tIntv(t) {}
			bool operator<(PatchPair const &o) const { return tIntv[0] > o.tIntv[0]; }
			double calcWidth() const{
				const double w1 = pb1.width(), w2 = pb2.width();
				return std::max(std::max(w1, w2), tIntv[1]-tIntv[0]);
			}
		};

		std::priority_queue<PatchPair> heap;

		ParamBound1 initParam1;
		ParamBound2 initParam2;
		Array2d initTimeIntv(0,upperTime);
		if (primitiveCheck(CpPos1, CpVel1, CpPos2, CpVel2, initParam1, initParam2, bb, initTimeIntv))
			heap.emplace(initParam1, initParam2, initTimeIntv);

		while (!heap.empty()) {
			PatchPair const cur = heap.top();
			heap.pop();

			// Meets the precision requirement
			if (cur.calcWidth() < deltaDist) {
				uv1 = cur.pb1.centerParam();
				uv2 = cur.pb2.centerParam();
				return cur.tIntv[0];
			}

			// Divide the time step into two halves
			double tMid = (cur.tIntv[0]+cur.tIntv[1])*0.5;
			Array2d divTime1(cur.tIntv[0],tMid), divTime2(tMid, cur.tIntv[1]);
			// Divide the current patch into two sets of four-to-four pieces
			for (int i = 0; i < 4; i++) {
				ParamBound1 divUvB1(cur.pb1.interpSubpatchParam(i));
				for (int j = 0; j < 4; j++) {
					ParamBound2 divUvB2(cur.pb2.interpSubpatchParam(j));
					if (primitiveCheck(CpPos1, CpVel1, CpPos2, CpVel2, divUvB1, divUvB2, bb, divTime1)){
						heap.emplace(divUvB1, divUvB2, divTime1);
					}
					if (primitiveCheck(CpPos1, CpVel1, CpPos2, CpVel2, divUvB1, divUvB2, bb, divTime2)){
						heap.emplace(divUvB1, divUvB2, divTime2);
					}
				}
			}
		}

		return -1;
	}
};