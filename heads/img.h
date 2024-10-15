
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
namespace Img
{
using namespace std;
// 获取原始图像
inline auto getOriImg(char c)
{
    char s[1005];
    s[0] = 0;
    char lk[128];
    std::vector<std::vector<bool>> img;
    int h = 65;
    int w = 38;
    lk['0'] = '0';
    lk['1'] = '1';
    lk['2'] = '2';
    lk['3'] = '3';
    lk['4'] = '4';
    lk['5'] = '5';
    lk['6'] = '6';
    lk['7'] = '7';
    lk['8'] = '8';
    lk['9'] = '9';
    lk['+'] = 'a';
    lk['-'] = 'b';
    lk['*'] = 'c';
    lk['/'] = 'd';
    lk['('] = 'e';
    lk[')'] = 'f';
    sprintf(s, "./img/%c.txt", lk[c]);
    freopen(s, "r", stdin);
    for (int i = 1; i <= h; i++)
    {
        scanf("%s", s + 1);
        img.push_back({});
        for (int j = 1; j <= w; j++)
        {
            if (s[j] == '#')
            {
                img[i - 1].push_back(true);
            }
            else
            {
                img[i - 1].push_back(false);
            }
        }
    }
    return img;
}
// 旋转图像
inline vector<vector<bool>> rotateImg(const vector<vector<bool>> &image, double R)
{
    int height = image.size();
    int width = image[0].size();
    vector<vector<bool>> rotated(height, vector<bool>(width, false));
    double radians = R * 3.141592653589793 / 180.0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int newX = static_cast<int>(round(x * cos(radians) - y * sin(radians)));
            int newY = static_cast<int>(round(x * sin(radians) + y * cos(radians)));
            if (newX >= 0 && newX < width && newY >= 0 && newY < height)
            {
                rotated[newY][newX] = image[y][x];
            }
        }
    }
    return rotated;
}
// 图像失真
inline vector<vector<bool>> distortImg(const vector<vector<bool>> &image, double Sx, double Sy)
{
    int height = image.size();
    int width = image[0].size();
    vector<vector<bool>> distorted(height, vector<bool>(width, false));
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int newX = static_cast<int>(round(x + Sy * y));
            int newY = static_cast<int>(round(Sx * x + y));
            if (newX >= 0 && newX < width && newY >= 0 && newY < height)
            {
                distorted[newY][newX] = image[y][x];
            }
        }
    }
    return distorted;
}
// 获取参数
inline void getArgs(int t, double &M, double &Mh, double &Mw, double &R, double &Sx, double &Sy, double &P)
{
    mt19937 gen(random_device{}());
    uniform_real_distribution<> disM(0.9, 1.0);
    uniform_real_distribution<> disMh(0.9, 1.0);
    uniform_real_distribution<> disMw(0.9, 1.0);
    uniform_real_distribution<> disR(-15.0, 15.0);
    uniform_real_distribution<> disSx(-0.1, 0.1);
    uniform_real_distribution<> disSy(-0.1, 0.1);
    uniform_real_distribution<> disP(0.0, 0.05);
    M = disM(gen);
    Mh = disMh(gen);
    Mw = disMw(gen);
    R = disR(gen);
    Sx = disSx(gen);
    Sy = disSy(gen);
    P = disP(gen);
    if (t < 30)
    {
        R = uniform_real_distribution<>(-2.0, 2.0)(gen);
        Sx = 0;
        Sy = 0;
        P = 0;
    }
    else if (t < 90)
    {
        R = uniform_real_distribution<>(-10.0, 10.0)(gen);
        P = 0.05;
    }
    else
    {
        R = uniform_real_distribution<>(-15.0, 15.0)(gen);
        P = 0.05;
    }
}
// 干扰图像
inline vector<vector<bool>> dirtyImg(const vector<vector<bool>> &original, int t = 100)
{
    double M, Mh, Mw, R, Sx, Sy, P;
    getArgs(t, M, Mh, Mw, R, Sx, Sy, P);
    int height = original.size();
    int width = original[0].size();
    int newHeight = static_cast<int>(round(height * M));
    int newWidth = static_cast<int>(round(width * M));
    vector<vector<bool>> resized(newHeight, vector<bool>(newWidth, false));
    for (int y = 0; y < newHeight; ++y)
    {
        for (int x = 0; x < newWidth; ++x)
        {
            int origX = static_cast<int>(round(x / Mw));
            int origY = static_cast<int>(round(y / Mh));
            if (origX >= 0 && origX < width && origY >= 0 && origY < height)
            {
                resized[y][x] = original[origY][origX];
            }
        }
    }
    vector<vector<bool>> rotated = rotateImg(resized, R);
    vector<vector<bool>> distorted = distortImg(rotated, Sx, Sy);
    return distorted;
}
// 获取带干扰图像
inline vector<vector<bool>> getDirtyImg(char original, int t = 100)
{
    return dirtyImg(getOriImg(original), t);
}
// 缩放图像
inline vector<vector<bool>> resizeImg(const vector<vector<bool>> &matrix, int h, int w)
{
    int originalHeight = matrix.size();
    int originalWidth = matrix[0].size();

    vector<vector<bool>> resizedMatrix(h, vector<bool>(w, false));

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            int origI = static_cast<int>(i * (originalHeight / static_cast<double>(h)));
            int origJ = static_cast<int>(j * (originalWidth / static_cast<double>(w)));
            resizedMatrix[i][j] = matrix[origI][origJ];
        }
    }

    return resizedMatrix;
}
// 剪切图像 - dfs
inline void trimDfs(const vector<vector<bool>> &matrix, int x, int y, vector<vector<bool>> &visited, int &minX,
                    int &minY, int &maxX, int &maxY)
{
    int rows = matrix.size();
    int cols = matrix[0].size();
    if (x < 0 || x >= rows || y < 0 || y >= cols || !matrix[x][y] || visited[x][y])
    {
        return;
    }
    visited[x][y] = true;
    minX = min(minX, x);
    minY = min(minY, y);
    maxX = max(maxX, x);
    maxY = max(maxY, y);
    trimDfs(matrix, x - 1, y, visited, minX, minY, maxX, maxY); // Up
    trimDfs(matrix, x + 1, y, visited, minX, minY, maxX, maxY); // Down
    trimDfs(matrix, x, y - 1, visited, minX, minY, maxX, maxY); // Left
    trimDfs(matrix, x, y + 1, visited, minX, minY, maxX, maxY); // Right
}
// 剪切图像
inline vector<vector<bool>> trimImg(const vector<vector<bool>> &matrix)
{
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(cols, false));
    int minX = rows, minY = cols, maxX = -1, maxY = -1;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (matrix[i][j] && !visited[i][j])
            {
                trimDfs(matrix, i, j, visited, minX, minY, maxX, maxY);
                break;
            }
        }
    }
    int newRows = maxX - minX + 1;
    int newCols = maxY - minY + 1;
    vector<vector<bool>> trimmedMatrix(newRows, vector<bool>(newCols, false));
    for (int i = minX; i <= maxX; ++i)
    {
        for (int j = minY; j <= maxY; ++j)
        {
            if (matrix[i][j])
            {
                trimmedMatrix[i - minX][j - minY] = true;
            }
        }
    }
    return trimmedMatrix;
}
// 将图片放在中间
inline vector<vector<bool>> centerImg(const vector<vector<bool>> &matrix, int h, int w)
{
    int originalHeight = matrix.size();
    int originalWidth = matrix.empty() ? 0 : matrix[0].size();

    int startRow = (h - originalHeight) / 2;
    int startCol = (w - originalWidth) / 2;

    vector<vector<bool>> newMatrix(h, vector<bool>(w, false));

    for (int i = 0; i < originalHeight; ++i)
    {
        for (int j = 0; j < originalWidth; ++j)
        {
            newMatrix[startRow + i][startCol + j] = matrix[i][j];
        }
    }
    return newMatrix;
}
// 获取标准图像
inline vector<vector<bool>> img(char c)
{
    auto v = getDirtyImg(c);
    v = trimImg(v);
    v = resizeImg(v, 32, 32);
    return v;
}
// 打印图像
inline void ptImg(const vector<vector<bool>> &v, FILE *f = stdout)
{
    for (const auto &row : v)
    {
        for (const auto &val : row)
        {
            fprintf(f, "%c", val ? '#' : '.');
        }
        fprintf(f, "\n");
    }
}
inline vector<bool> getBit(const vector<vector<bool>> &v)
{
    vector<bool> r;
    for (const auto &row : v)
    {
        for (const auto &val : row)
        {
            r.push_back(val);
        }
    }
    return r;
}
} // namespace Img