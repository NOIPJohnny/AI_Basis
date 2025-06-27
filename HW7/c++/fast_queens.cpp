#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include "utils/random_variables.hpp"
class Queen {
private:
    int num;
    std::vector<int> queen_array;
    std::vector<int> Uconflict_array;
    std::vector<int> Dconflict_array;

public:
    Queen(int n) : num(n) {
        
        queen_array.resize(num + 1);
        Uconflict_array.resize(2 * num + 1, 0);
        Dconflict_array.resize(2 * num + 1, 0);
        
 
        for (int i = 1; i <= num; i++) {
            queen_array[i] = i;
            

            Uconflict_array[i + i] += 1;           
            Dconflict_array[num + i - i] += 1;     
        }
    }


    inline void N_update(int col1, int col2) {
        int row1 = queen_array[col1];
        int row2 = queen_array[col2];

        int tDown = num + col1 - row1;
        Dconflict_array[tDown] -= 1;
        Uconflict_array[col1 + row1] -= 1;

        tDown = num + col2 - row2;
        Dconflict_array[tDown] -= 1;
        Uconflict_array[col2 + row2] -= 1;

        queen_array[col1] = row2;
        queen_array[col2] = row1;

        tDown = num + col1 - queen_array[col1];
        Dconflict_array[tDown] += 1;
        Uconflict_array[col1 + queen_array[col1]] += 1;

        tDown = num + col2 - queen_array[col2];
        Dconflict_array[tDown] += 1;
        Uconflict_array[col2 + queen_array[col2]] += 1;
    }

    inline int N_Lconflict(int col1, int col2) {
        int row = queen_array[col2];
        int tDown = num + col1 - row;
        return Uconflict_array[row + col1] + Dconflict_array[tDown];
    }

    int init_search() {
        int j = 1;
        const double SEARCH_FACTOR = 2.5;
        
        for (int i = 1; i < int(SEARCH_FACTOR * num); i++) {
            int m = j + (RandomVariables::uniform_int() % (num - j + 1));
            if (N_Lconflict(j, m) == 0) {
                N_update(j, m);
                j++;
                if (j == num) {
                    return 0;
                }
            }
        }
        
        for (int i = j; i <= num; i++) {
            int m = i + (RandomVariables::uniform_int() % (num - i + 1));
            N_update(i, m);
        }
        
        return num - j + 1;
    }

    inline int total_conflict(int col) {
        int row = queen_array[col];
        int tDown = num + col - row;
        return Uconflict_array[col + row] + Dconflict_array[tDown] - 2;
    }

    void final_search(int k) {
        std::vector<int> conflict_cols;
        conflict_cols.reserve(k + 1);

        for (int i = num - k + 1; i <= num; i++) {
            if (total_conflict(i) > 0) {
                conflict_cols.push_back(i);
            }
        }
        
        while (!conflict_cols.empty()) {
            std::vector<int> current_conflicts = conflict_cols;
            conflict_cols.clear();
            
            for (int i : current_conflicts) {
                if (total_conflict(i) > 0) {
                    int max_attempts = 50;
                    
                    for (int attempt = 0; attempt < max_attempts; attempt++) {
                        int j = 1 + (RandomVariables::uniform_int() % num);
                        N_update(i, j);
                        
                        if ((total_conflict(i) > 0) || (total_conflict(j) > 0)) {
                            N_update(i, j); 
                        } else {
                            break;
                        }
                        
                        if (attempt == max_attempts - 1 && total_conflict(i) > 0) {
                            conflict_cols.push_back(i);
                        }
                    }
                }
            }
            
            if (conflict_cols.size() >= current_conflicts.size()) {

                std::vector<int> indices = RandomVariables::uniform_permutation(num);
                for (int i = 0; i < num/100; i++) {
                    int col1 = indices[i] + 1; 
                    int col2 = indices[num-i-1] + 1;
                    if (col1 > 0 && col1 <= num && col2 > 0 && col2 <= num) {
                        N_update(col1, col2);
                    }
                }
            }
            
            if (conflict_cols.size() > k * 2) {
                break;
            }
        }
    }
    
    bool quick_verify(int sample_size = 5000) {

        std::vector<int> indices = RandomVariables::uniform_permutation(num);
        
        for (int i = 0; i < std::min(sample_size, num); i++) {
            int col = indices[i] % num + 1;
            if (total_conflict(col) > 0) {
                return false;
            }
        }
        return true;
    }
    
    bool verify_solution() {
        for (int i = 1; i <= num; i++) {
            if (total_conflict(i) > 0) {
                return false;
            }
        }
        return true;
    }
    
    void print_solution(bool to_file = false) {
        if (to_file) {
            std::ofstream outfile("queens_solution.txt");
            if (outfile.is_open()) {
                for (int i = 1; i <= num; i++) {
                    outfile << queen_array[i] << std::endl;
                }
                outfile.close();
                std::cout << "Solution saved to queens_solution.txt" << std::endl;
            }
        } else {
            std::cout << "First 20 queen positions: ";
            for (int i = 1; i <= std::min(20, num); i++) {
                std::cout << queen_array[i] << " ";
            }
            std::cout << std::endl;
        }
    }
};

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int n = 4000000;
    std::cout << "Solving " << n << "-Queens problem..." << std::endl;
    
    Queen queen(n);
    
    auto init_start = std::chrono::high_resolution_clock::now();
    int conflicts = queen.init_search();
    auto init_end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Initial search completed with " << conflicts 
              << " conflicts remaining in " 
              << std::chrono::duration<double>(init_end - init_start).count()
              << " seconds" << std::endl;
    
    auto final_start = std::chrono::high_resolution_clock::now();
    queen.final_search(conflicts);
    auto final_end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Final search completed in " 
              << std::chrono::duration<double>(final_end - final_start).count()
              << " seconds" << std::endl;
    
    
    auto end_time = std::chrono::high_resolution_clock::now();
    bool quick_valid = queen.quick_verify();
    std::cout << "Quick validation: solution appears " << (quick_valid ? "valid" : "invalid") << std::endl;
    
    if (quick_valid) {
        std::cout << "Performing complete validation..." << std::endl;
        bool valid = queen.verify_solution();
        std::cout << "Complete validation: solution is " << (valid ? "valid" : "invalid") << std::endl;
    }
    
    
    double total_duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total time: " << total_duration << " seconds" << std::endl;
    
    if (quick_valid) {
        queen.print_solution(true);
    }
    
    return 0;
}