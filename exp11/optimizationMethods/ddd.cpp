//
// Created by Vahe Aleksanyan on 06.05.24.
//

#include <vector>
#include <map>
#include <iostream>

int sumOfUnique(const std::vector<int> & nums) {
    std::map<int, int> mp;
    int sum = 0;

    for(int num : nums) {
        mp[num]++;
    }

    for(const auto& pair : mp) {
        if(pair.second == 1) {
            sum+= pair.first;
        }
    }
    return sum;
}
int main() {
    std::vector<int> nums = {1,2,2,2,4, 5, 1};

    int result = sumOfUnique(nums);

    std::cout << "the result is " << result << std::endl;
}