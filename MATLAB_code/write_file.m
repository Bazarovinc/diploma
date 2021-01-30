clear all
hbar=1.06e-34; % Постоянная Дирака
q=1.6e-19;  % Заряд электрона
m_t = .25*9.1e-31; % Тестовое значения эффективной массы
Ef_t = 0.1; % Тестовое значение уровня Ферми
a_t = 3e-10; % Тестовое значение шага
% Создание массива тренировочных значений эффективной массы
m = 0.250*9.1e-31:0.001*9.1e-31:0.300*9.1e-31;
% Создание массива тренировчных значений уровня Ферми
Ef = 0.050:0.001:0.110;
% Создание массива тренировчных значений шага
a = 2.90e-10:0.01e-10:3.05e-10;
NS=15; % Число узлов в 1-ом контакте
NC=16; % Число узлов в канале
ND=15; % Число узлов в 2-ом контакте
Np=NS+NC+ND; % Число узлов во всей структуре
% Задание значений величин энергии
lev = [0.2, 0.3, 0.4, 0.5];
U_c = [];
U_t = [];
% Создание массивов Резонансно-тунельная структура
for i=1:length(lev)
    % Массив со значениями для создания дата сета
    U_c(i,:) = [zeros(NS,1);lev(i)*ones(4,1);zeros(NC-8,1);lev(i)*ones(4,1);zeros(ND,1)];
    % Массив со значениями для тренировки ИНС
    U_t(i,:) = [0.01*ones(NS,1);lev(i)*ones(4,1);0.01*ones(NC-8,1);lev(i)*ones(4,1);0.01*ones(ND,1)];
end
l = 0;
% Открытие файлов для записи входных данных для тренировки ИНС
fd_1 = fopen('C:\Users\nikve\Desktop\Диплом\data_sets\input_test.csv','a');
% Открытие файлов для записи выходных данных для тренировки ИНС
fd_2 = fopen('C:\Users\nikve\Desktop\Диплом\data_sets\output_test.csv', 'a');
% Цикл генерации тренирочного дата сета
for i=1:length(m)
    for j=1:length(Ef)
        for k=1:length(a)
            for n=1:length(lev)
                if m(i) == 1 && Ef(j) == 1 && a(k) == 1 && lev(n) == 1
                else
                    l
                    l = l + 1;
                    t0=(hbar^2)/(2*m(i)*(a(k)^2)*q);
                    UB = U_c(n, :);
                    U = U_t(n, :);
                    H=(2*t0*diag(ones(1,Np)))-(t0*diag(ones(1,Np-1),1))-(t0*diag(ones(1,Np-1),-1));
                    H=H+diag(UB);
                    % Получение тренировочных значений тока
                    I = create_I(m(i), Ef(j), a(k), NS, NC, ND, Np, UB, H) * 1e5;
                    % Запись в формат для записи в файл
                    input = {[m(i)*1e30], [Ef(j)], [a(k)* 1e9], U};
                    output = {I};
                    % Запись данных в файлы
                    fprintf(fd_1,'%f,', input{1:end});
                    fprintf(fd_1, '\n');
                    fprintf(fd_2,'%f,', output{1:end});
                    fprintf(fd_2, '\n');
                end
            end
        end
    end
end
% Закрытие файлов
fclose(fd_1);
fclose(fd_2);