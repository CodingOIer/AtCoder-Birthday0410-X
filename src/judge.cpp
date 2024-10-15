#include "heads/img.h"
#include "heads/net.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
constexpr int CalcCount = 1000;
std::mt19937 rnd(std::random_device{}());
std::vector<bool> get(char ch)
{
    return Img::getBit(Img::img(ch));
}
int main(int, char *argc[])
{
    const char *list = "0123456789+-*/()";
    auto gen = Net::read(argc[1]);
    double ac = 1;
    double fc = 0;
    for (int i = 0; i < 16; i++)
    {
        double less;
        double most;
        less = most = 0;
        char ch = list[i];
        double cl;
        double cm;
        cl = cm = 0;
        for (int j = 1; j <= CalcCount; j++)
        {
            auto t = gen.generate(get(ch));
            for (int k = 0; k < 16; k++)
            {
                if (i == k)
                {
                    cm += most;
                    most = std::max(most, t[k]);
                }
                else
                {
                    cl += less;
                    less = std::max(less, t[k]);
                }
            }
        }
        cm /= CalcCount;
        cl /= CalcCount * 15;
        ac = std::min(ac, most - less);
        fc += (cm - cl) * (cm - cl);
        printf("%c: for the true, most is %.12lf, less if %.12lf\n", ch, most, less);
    }
    printf("The danger is %.12lf, score is %.12lf\n", ac, fc / 16 * 1000);
    return 0;
}