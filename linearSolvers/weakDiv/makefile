#.......................................................#
#.
#.Makefile to compile weak division Eigen+CUDA program
#.Manaure Francisquez.
#.November 2019.
#.
#.......................................................#

COMPILER = nvcc

PROGNAME = weakDiv
PROGEXT  = cpp

EIGEN_INC = $(HOME)/gkylsoft/eigen3/include/eigen3/
EIGEN_LIB = 

CUDA_INC  = 
CUDA_LIB  = -lcublas

INCS      = -I $(EIGEN_INC)
LIBS      = $(CUDA_LIB)

FLAGS = 

clean: $(PROGNAME).x
	rm -f $(addsuffix .mod, $(mods))

$(PROGNAME).x: $(PROGNAME).$(PROGEXT)
	$(COMPILER) $(INCS) $(LIBS) $(FLAGS) weakDiv_Eigen.cpp weakDiv_CUDA.cu $(PROGNAME).$(PROGEXT) -o $(PROGNAME).x 
#	$(COMPILER) $(INCS) $(LIBS) $(FLAGS) weakDiv_CUDA.cu $(PROGNAME).$(PROGEXT) -o $(PROGNAME).x 

$(PROGNAME):
	echo 'Error: You need the .x extension'
