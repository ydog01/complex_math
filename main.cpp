#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <complex>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include <functional>
#include "eval.hpp"

using namespace std;

const double pi = 3.14159265358979323846;

struct TextInputBox
{
    SDL_Rect rect;
    string text;
    bool active;
    bool visible;
};

struct UIButton
{
    SDL_Rect rect;
    string label;
    bool visible;
    function<void()> action;
};

struct Point
{
    complex<double> original;
    complex<double> transformed;
};

struct AppState
{
    function<complex<double>(complex<double>)> compute_func;
    string current_expr = "gamma(z)";
    bool expr_valid = true;
    Uint32 animation_start = 0;
    bool show_error = false;
    string error_msg;
    vector<vector<Point>> verticalLines;
    vector<vector<Point>> horizontalLines;
    bool enable_antialiasing = true;
};

complex<double> complex_gamma(complex<double> z);
shared_ptr<sstree<eval::var<complex<double>>>> init_vars();
shared_ptr<sstree<eval::func<complex<double>>>> init_funcs();
shared_ptr<sstree<eval::func<complex<double>>>> init_operators();

class EvaluatorWrapper
{
public:
    eval::eval<complex<double>> instance;

    EvaluatorWrapper() : instance(
                             [](const char &c)
                             { return isdigit(c) || c == '.'; },
                             [](const char &c)
                             { return isdigit(c) || c == '.'; },
                             [](const string &s)
                             { return complex<double>(stod(s)); },
                             init_vars(),
                             init_funcs(),
                             nullptr,
                             init_operators()) {}
};

EvaluatorWrapper &get_evaluator()
{
    static EvaluatorWrapper evaluator;
    return evaluator;
}

AppState app_state;

complex<double> complex_gamma(complex<double> z)
{
    static const double g = 7.0;
    static const double p[] = {
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

    if (z.real() < 0.5)
        return pi / (sin(pi * z) * complex_gamma(1.0 - z));
    z -= 1.0;
    complex<double> x = p[0];
    for (int i = 1; i < 9; ++i)
        x += p[i] / (z + complex<double>(i));
    complex<double> t = z + g + 0.5;
    return sqrt(2 * pi) * pow(t, z + 0.5) * exp(-t) * x;
}

shared_ptr<sstree<eval::var<complex<double>>>> init_vars()
{
    static auto vars = make_shared<sstree<eval::var<complex<double>>>>();
    static bool initialized = false;

    if (!initialized)
    {
        vars->insert("z", {eval::vartype::FREEVAR, complex<double>(0)});
        vars->insert("i", {eval::vartype::CONSTVAR, complex<double>(0, 1)});
        vars->insert("pi", {eval::vartype::CONSTVAR, complex<double>(M_PI)});
        vars->insert("e", {eval::vartype::CONSTVAR, complex<double>(exp(1.0))});
        initialized = true;
    }
    return vars;
}

shared_ptr<sstree<eval::func<complex<double>>>> init_funcs()
{
    static auto funcs = make_shared<sstree<eval::func<complex<double>>>>();
    static bool initialized = false;

    if (!initialized)
    {
        funcs->insert("sin", {1, 10, [](const complex<double> *args)
                              { return sin(args[0]); }});
        funcs->insert("cos", {1, 10, [](const complex<double> *args)
                              { return cos(args[0]); }});
        funcs->insert("tan", {1, 10, [](const complex<double> *args)
                              { return tan(args[0]); }});
        funcs->insert("arcsin", {1, 10, [](const complex<double> *args)
                                 { return asin(args[0]); }});
        funcs->insert("arccos", {1, 10, [](const complex<double> *args)
                                 { return acos(args[0]); }});
        funcs->insert("arctan", {1, 10, [](const complex<double> *args)
                                 { return atan(args[0]); }});
        funcs->insert("sh", {1, 10, [](const complex<double> *args)
                             { return sinh(args[0]); }});
        funcs->insert("ch", {1, 10, [](const complex<double> *args)
                             { return cosh(args[0]); }});
        funcs->insert("th", {1, 10, [](const complex<double> *args)
                             { return tanh(args[0]); }});
        funcs->insert("arsh", {1, 10, [](const complex<double> *args)
                               { return asinh(args[0]); }});
        funcs->insert("arch", {1, 10, [](const complex<double> *args)
                               { return acosh(args[0]); }});
        funcs->insert("arth", {1, 10, [](const complex<double> *args)
                               { return atanh(args[0]); }});
        funcs->insert("gamma", {1, 10, [](const complex<double> *args)
                                { return complex_gamma(args[0]); }});
        initialized = true;
    }
    return funcs;
}

shared_ptr<sstree<eval::func<complex<double>>>> init_operators()
{
    static auto oper = make_shared<sstree<eval::func<complex<double>>>>();
    static bool initialized = false;

    if (!initialized)
    {
        oper->insert("+", {2, 2, [](const complex<double> *args)
                           { return args[0] + args[1]; }});
        oper->insert("-", {2, 2, [](const complex<double> *args)
                           { return args[0] - args[1]; }});
        oper->insert("*", {2, 3, [](const complex<double> *args)
                           { return args[0] * args[1]; }});
        oper->insert("/", {2, 3, [](const complex<double> *args)
                           { return args[0] / args[1]; }});
        oper->insert("^", {2, 4, [](const complex<double> *args)
                           { return pow(args[0], args[1]); }});
        initialized = true;
    }
    return oper;
}

void init_default_func()
{
    app_state.compute_func = [](complex<double> z)
    {
        return complex_gamma(z);
    };
}

void generate_data(const function<complex<double>(complex<double>)> &func)
{
    const double x_halfRange = 4.0;
    const double y_halfRange = x_halfRange / (1920.0 / 1080.0);

    app_state.verticalLines.clear();
    app_state.horizontalLines.clear();

    for (double x = -x_halfRange; x <= x_halfRange; x += 0.25)
    {
        vector<Point> line;
        for (double y = -y_halfRange; y <= y_halfRange; y += 0.05)
        {
            complex<double> z(x, y);
            line.push_back({z, func(z)});
        }
        app_state.verticalLines.push_back(line);
    }

    for (double y = -y_halfRange; y <= y_halfRange; y += 0.25)
    {
        vector<Point> line;
        for (double x = -x_halfRange; x <= x_halfRange; x += 0.05)
        {
            complex<double> z(x, y);
            line.push_back({z, func(z)});
        }
        app_state.horizontalLines.push_back(line);
    }
}

void parse_expression()
{
    try
    {
        eval::epre<complex<double>> expr;
        size_t errpos = get_evaluator().instance.cpre(expr, app_state.current_expr);

        if (errpos != (size_t)-1)
        {
            throw runtime_error("语法错误，位置: " + to_string(errpos));
        }

        app_state.compute_func = [expr](complex<double> z)
        {
            auto var_node = init_vars()->search("z");
            if (var_node && var_node->data)
                var_node->data->value = z;

            try
            {
                return get_evaluator().instance.result(expr);
            }
            catch (...)
            {
                return complex<double>(NAN, NAN);
            }
        };

        app_state.expr_valid = true;
        app_state.animation_start = SDL_GetTicks();
        app_state.show_error = false;

        generate_data(app_state.compute_func);
    }
    catch (const exception &e)
    {
        app_state.expr_valid = false;
        app_state.show_error = true;
        app_state.error_msg = "错误: " + string(e.what());
        init_default_func();
        generate_data(app_state.compute_func);
    }
}

SDL_Point complexToPixel(complex<double> z, double scale, int centerX, int centerY)
{
    const double maxVal = 1e10;
    if (abs(z.real()) > maxVal || abs(z.imag()) > maxVal)
    {
        return {-10000, -10000};
    }
    int x = centerX + static_cast<int>(z.real() * scale);
    int y = centerY - static_cast<int>(z.imag() * scale);
    return {x, y};
}

SDL_Color HSVtoRGB(float H, float S, float V)
{
    float C = V * S;
    float X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
    float m = V - C;
    float r, g, b;

    if (H >= 0 && H < 60)
        r = C, g = X, b = 0;
    else if (H >= 60 && H < 120)
        r = X, g = C, b = 0;
    else if (H >= 120 && H < 180)
        r = 0, g = C, b = X;
    else if (H >= 180 && H < 240)
        r = 0, g = X, b = C;
    else if (H >= 240 && H < 300)
        r = X, g = 0, b = C;
    else
        r = C, g = 0, b = X;

    return {static_cast<Uint8>((r + m) * 255),
            static_cast<Uint8>((g + m) * 255),
            static_cast<Uint8>((b + m) * 255), 255};
}

void drawAALine(SDL_Renderer *renderer, int x0, int y0, int x1, int y1, SDL_Color color)
{
    auto plot = [&](int x, int y, float alpha)
    {
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, static_cast<Uint8>(alpha * 255));
        SDL_RenderDrawPoint(renderer, x, y);
    };

    bool steep = abs(y1 - y0) > abs(x1 - x0);
    if (steep)
        swap(x0, y0), swap(x1, y1);
    if (x0 > x1)
        swap(x0, x1), swap(y0, y1);

    float dx = x1 - x0;
    float dy = y1 - y0;
    float gradient = (dx == 0) ? 1.0f : dy / dx;

    float xend = round(x0);
    float yend = y0 + gradient * (xend - x0);
    float xgap = 1 - fmod(x0 + 0.5f, 1);
    int xpxl1 = xend;
    int ypxl1 = floor(yend);
    if (steep)
    {
        plot(ypxl1, xpxl1, (1 - fmod(yend, 1)) * xgap);
        plot(ypxl1 + 1, xpxl1, fmod(yend, 1) * xgap);
    }
    else
    {
        plot(xpxl1, ypxl1, (1 - fmod(yend, 1)) * xgap);
        plot(xpxl1, ypxl1 + 1, fmod(yend, 1) * xgap);
    }

    float intery = yend + gradient;

    xend = round(x1);
    yend = y1 + gradient * (xend - x1);
    xgap = fmod(x1 + 0.5f, 1);
    int xpxl2 = xend;
    int ypxl2 = floor(yend);
    if (steep)
    {
        plot(ypxl2, xpxl2, (1 - fmod(yend, 1)) * xgap);
        plot(ypxl2 + 1, xpxl2, fmod(yend, 1) * xgap);
    }
    else
    {
        plot(xpxl2, ypxl2, (1 - fmod(yend, 1)) * xgap);
        plot(xpxl2, ypxl2 + 1, fmod(yend, 1) * xgap);
    }

    if (steep)
    {
        for (int x = xpxl1 + 1; x < xpxl2; x++)
        {
            plot(floor(intery), x, 1 - fmod(intery, 1));
            plot(floor(intery) + 1, x, fmod(intery, 1));
            intery += gradient;
        }
    }
    else
    {
        for (int x = xpxl1 + 1; x < xpxl2; x++)
        {
            plot(x, floor(intery), 1 - fmod(intery, 1));
            plot(x, floor(intery) + 1, fmod(intery, 1));
            intery += gradient;
        }
    }
}

void draw_grid(SDL_Renderer *renderer, const vector<vector<Point>> &lines, double t)
{
    const double scale = 1920.0 / 8.0;
    const int centerX = 1920 / 2;
    const int centerY = 1080 / 2;
    const int screenMargin = 500;

    auto complexToPixel = [&](complex<double> z) -> SDL_Point
    {
        int x = centerX + static_cast<int>(z.real() * scale);
        int y = centerY - static_cast<int>(z.imag() * scale);
        return {x, y};
    };

    auto lerp = [t](complex<double> a, complex<double> b)
    {
        double smooth_t = t * t * (3 - 2 * t);
        return a + smooth_t * (b - a);
    };

    for (const auto &line : lines)
    {
        vector<SDL_Point> visiblePoints;

        for (size_t i = 0; i < line.size(); ++i)
        {
            complex<double> current = lerp(line[i].original, line[i].transformed);
            SDL_Point p = complexToPixel(current);

            if (p.x < -screenMargin || p.x > 1920 + screenMargin ||
                p.y < -screenMargin || p.y > 1080 + screenMargin)
            {
                continue;
            }

            visiblePoints.push_back(p);
        }

        if (visiblePoints.size() >= 2)
        {
            double angle = arg(line.front().transformed) * 180 / pi;
            SDL_Color color = HSVtoRGB(fmod(angle + 360, 360), 1.0, 1.0);

            if (app_state.enable_antialiasing)
            {

                for (size_t i = 0; i < visiblePoints.size() - 1; ++i)
                {
                    drawAALine(renderer, visiblePoints[i].x, visiblePoints[i].y,
                               visiblePoints[i + 1].x, visiblePoints[i + 1].y, color);
                }
            }
            else
            {

                SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 255);
                SDL_RenderDrawLines(renderer, visiblePoints.data(), visiblePoints.size());
            }
        }
    }
}

#undef main
#undef main
int main()
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        cerr << "SDL初始化失败: " << SDL_GetError() << endl;
        return 1;
    }

    if (TTF_Init() != 0)
    {
        cerr << "TTF初始化失败: " << TTF_GetError() << endl;
        SDL_Quit();
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("复变函数可视化",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1920, 1080, 0);
    if (!window)
    {
        cerr << "窗口创建失败: " << SDL_GetError() << endl;
        TTF_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1,
                                                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer)
    {
        cerr << "渲染器创建失败: " << SDL_GetError() << endl;
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }

    TTF_Font *font = TTF_OpenFont("msyh.ttc", 24);
    if (!font)
    {
        cerr << "字体加载失败: " << TTF_GetError() << endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }

    TTF_Font *bold_font = TTF_OpenFont("msyhbd.ttc", 24);
    if (!bold_font)
    {
        cerr << "粗体字体加载失败: " << TTF_GetError() << endl;
        TTF_CloseFont(font);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }

    init_default_func();
    generate_data(app_state.compute_func);

    TextInputBox input_box = {{50, 1080 - 70, 600, 40}, "gamma(z)", false, true};
    UIButton run_btn = {{660, 1080 - 70, 80, 40}, "运行", true, parse_expression};
    UIButton aa_btn = {{750, 1080 - 70, 145, 40}, "抗锯齿: 开", true, []
                       {
                           app_state.enable_antialiasing = !app_state.enable_antialiasing;
                       }};
    UIButton quit_btn = {{1920 - 110, 1080 - 70, 70, 40}, "退出", true, []
                         {
                             SDL_Event evt;
                             evt.type = SDL_QUIT;
                             SDL_PushEvent(&evt);
                         }};

    Uint32 lasttime;
    bool running = true;
    SDL_StartTextInput();
    while (running)
    {
        lasttime = SDL_GetTicks();
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = false;
            }

            if (event.type == SDL_TEXTINPUT && input_box.active)
            {
                input_box.text += event.text.text;
                app_state.current_expr = input_box.text;
            }

            if (event.type == SDL_KEYDOWN && input_box.active)
            {
                if (event.key.keysym.sym == SDLK_BACKSPACE && !input_box.text.empty())
                {
                    input_box.text.pop_back();
                    app_state.current_expr = input_box.text;
                }
            }

            if (event.type == SDL_MOUSEBUTTONDOWN)
            {
                int x = event.button.x, y = event.button.y;

                input_box.active = (x >= input_box.rect.x && x <= input_box.rect.x + input_box.rect.w &&
                                    y >= input_box.rect.y && y <= input_box.rect.y + input_box.rect.h);

                auto check_btn_click = [x, y](UIButton &btn)
                {
                    if (btn.visible && x >= btn.rect.x && x <= btn.rect.x + btn.rect.w &&
                        y >= btn.rect.y && y <= btn.rect.y + btn.rect.h)
                    {
                        btn.action();
                        return true;
                    }
                    return false;
                };

                check_btn_click(run_btn);
                check_btn_click(aa_btn);
                check_btn_click(quit_btn);
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        double t = min((SDL_GetTicks() - app_state.animation_start) / 5000.0, 1.0);

        draw_grid(renderer, app_state.verticalLines, t);
        draw_grid(renderer, app_state.horizontalLines, t);

        auto draw_text = [&](int x, int y, const string &text, SDL_Color color, TTF_Font *font)
        {
            SDL_Surface *surf = TTF_RenderUTF8_Blended(font, text.c_str(), color);
            if (!surf)
                return;

            SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
            if (tex)
            {
                SDL_Rect dst = {x, y, surf->w, surf->h};
                SDL_RenderCopy(renderer, tex, NULL, &dst);
                SDL_DestroyTexture(tex);
            }
            SDL_FreeSurface(surf);
        };

        SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
        SDL_RenderFillRect(renderer, &input_box.rect);
        SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
        SDL_RenderDrawRect(renderer, &input_box.rect);
        draw_text(input_box.rect.x + 10, input_box.rect.y + 3, input_box.text, {255, 255, 255}, font);

        auto draw_button = [&](UIButton &btn)
        {
            if (!btn.visible)
                return;

            SDL_SetRenderDrawColor(renderer, 80, 80, 80, 255);
            SDL_RenderFillRect(renderer, &btn.rect);
            SDL_SetRenderDrawColor(renderer, 150, 150, 150, 255);
            SDL_RenderDrawRect(renderer, &btn.rect);

            draw_text(btn.rect.x + 10, btn.rect.y + 5, btn.label, {255, 255, 255}, bold_font);
        };

        aa_btn.label = app_state.enable_antialiasing ? "抗锯齿: 开" : "抗锯齿: 关";

        draw_button(run_btn);
        draw_button(aa_btn);
        draw_button(quit_btn);

        if (app_state.show_error)
        {
            SDL_Color red = {255, 50, 50};
            draw_text(50, 1080 - 120, app_state.error_msg, red, font);
        }

        SDL_RenderPresent(renderer);

        if (SDL_GetTicks() - lasttime < 17)
            SDL_Delay(17 - SDL_GetTicks() + lasttime);
    }

    TTF_CloseFont(font);
    TTF_CloseFont(bold_font);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();
    return 0;
}