#include "scene_random.h"
#include "scene_examples.h"
#include "argsParser.h"
#include "cuda/myEigen.cuh"

inline std::unique_ptr<ArgsParser> BuildArgsParser()
{
	auto parser = std::make_unique<ArgsParser>();
	parser->addArgument<std::string>("solver", 's', "type of ccd solver (trad, td)", "td");
	parser->addArgument<std::string>("experiment", 'e', "type of experiment (rand, single, bunny)", "rand");
	parser->addArgument<std::string>("bb", 'b', "type of bounding box (aabb, obb)", "obb");

	parser->addArgument<double>("delta", 'd', "distance for convergence criterion", 1e-5);
	parser->addArgument<int>("kase", 'k', "number of generated cases", 100);
	parser->addArgument<int>("nvda", 'n', "use nvidia gpu mode", 0);
	return parser;
}

int main(int argc, char *argv[]){
	const int N = 5;
	double a[5] = {1,2,3,4,5};
	double b[5] = {0,5,10,15,20};
	cpy(b, a, N);
	std::cout << "Results:\n";
    // for (int i = 0; i < N; ++i) {
    //     std::cout << a[i] << " | " << a[i] << std::endl;
    // }

	auto parser = BuildArgsParser();
	parser->parse(argc, argv);

    const auto expType = std::any_cast<std::string>(parser->getValueByName("experiment"));	
    const auto solverType = std::any_cast<std::string>(parser->getValueByName("solver"));	
    const auto bbType = std::any_cast<std::string>(parser->getValueByName("bb"));
    const auto deltaDist = std::any_cast<double>(parser->getValueByName("delta"));
    const auto kase = std::any_cast<int>(parser->getValueByName("kase"));
	const auto nvda = std::any_cast<int>(parser->getValueByName("nvda"));

	BoundingBoxType bb;
	if(bbType=="obb")
		bb = BoundingBoxType::OBB;
	else if(bbType=="aabb")
		bb = BoundingBoxType::AABB;
	else{
		std::cerr<<"Bounding box not implemented.\n";
		exit(-1);
	}
	SolverType solver;
	if(solverType=="td")
		solver = SolverType::TDIntv;
	else if(solverType=="trad")
		solver = SolverType::TradIntv;
	else{
		std::cerr<<"Solver not implemented.\n";
		exit(-1);
	}

	if(expType=="rand")
		if(nvda==0) randomTest<RecCubicBezier, RecParamBound>(solver, bb, deltaDist, kase);
		else randomTestGPU(solver, bb, deltaDist, kase);
	else if(expType=="single")
		singleTest(solver, bb, deltaDist);
	else if(expType=="bunny")
		parabolaBunnyTorus(solver, bb, deltaDist);
	else{
		std::cerr<<"Experiment not implemented.\n";
		exit(-1);
	}
}