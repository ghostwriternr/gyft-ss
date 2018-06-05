PHONY  := all clean
TARGET := gyft-ss

SRC_DIR   := src
CPP_FILES := $(wildcard $(SRC_DIR)/*.cpp)

OBJ_DIR   := obj
OBJ_FILES := $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))

CXX          := g++
WARNINGS     := -Wall -Wextra -ansi -pedantic
INCLUDE_DIRS := -I./include
CXXFLAGS     := $(WARNINGS) $(INCLUDE_DIRS) `pkg-config --cflags opencv lept tesseract` -O -std=c++11
LDLIBS       := `pkg-config --libs opencv lept tesseract`

RM    := rm -f
RMDIR := rmdir

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CXX) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJ_DIR):
	@mkdir -p $@

clean:
	$(RM) $(OBJ_DIR)/*.o $(TARGET)
	$(RMDIR) $(OBJ_DIR)