all:
	g++ -c main.cpp Data.cpp Feature.cpp KNN.cpp EnsembleKNN.cpp shuffleData.cpp
	g++ -o main main.o Data.o Feature.o KNN.o EnsembleKNN.o shuffleData.o
	./main
clean:
	del /f main.o Data.o Feature.o KNN.o EnsembleKNN.o
	del /f main.exe	