#include "heads/img.h"
#include "heads/net.h"
#include <cstdio>
#include <random>
#include <vector>
std::mt19937 rnd(std::random_device{}());
Net::NeuralNetworkTrainer tra;
int main()
{
    const char *list = "0123456789+-*/()";
    double lr = 0.5;
    for (int i = 1; i <= 1e7; i++)
    {
        int id = rnd() % 16;
        char k = list[id];
        std::vector<double> target(1024, 0);
        target[id] = 1;
        auto v = Img::getBit(Img::img(k));
        tra.train({v}, {target}, 1, lr);
        printf("Train case %d, char is %c, lr is %.8lf\n", i, k, lr);
        if (i >> 12 << 12 == i)
        {
            Net::save(tra, "./data.txt");
        }
        lr *= 0.999995;
    }
    Net::save(tra, "./data.txt");
    return 0;
}