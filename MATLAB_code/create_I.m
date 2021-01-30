function II = create_I(m, Ef, a, NS, NC, ND, Np, UB, H)
    % Константы
    hbar=1.06e-34; % Постоянная Дирака
    q=1.6e-19; % Заряд электрона
    IE=(q^2)/(2*pi*hbar); % Коэффициент при I
    kT=.025; % Энергия теплового движения
    t0=(hbar^2)/(2*m*(a^2)*q); % Коэффеиент в Гамильтониане
    % Сетка из значений потенциала
    NV=30;
    VV=linspace(0,.5,NV);
    for iV=1:NV
       V = VV(iV);
        mu1 = Ef + (V/2);
        mu2 = Ef - (V/2);
        U = V*[0.5*ones(1, NS) linspace(0.5, -0.5, NC) -0.5*ones(1, ND)];
        % Применяемый потенциальный профиль
        U = U';
        % Сетка из значений энергий
           NE = 101;
        E = linspace(-0.2, 0.8, NE);
        % Разница между двумя уровнями
        dE = E(2) - E(1);
        f1 = 1./(1 + exp((E - mu1)./kT));
        f2 = 1./(1 + exp((E - mu2)./kT));
        % *****************___Прозрачность___*****************
        % Ток
        I = 0;
        for k = 1:NE
            % Граничные условия (функции отражения от перехода контакт-канал)
            sig1 = zeros(Np);
            sig2 = zeros(Np); 
            zplus = 1i*1e-12;
            % Нахождение cos(ka)
            ck = 1 - ((E(k) - U(1) - UB(1))/(2*t0));
            % ka = acos(ka)
            ka = acos(ck);
            sig1(1, 1) = -t0*exp(1i*ka);
            % Уширение
            gam1 = 1i*(sig1 - sig1');
            % Аналогично
            ck = 1 - ((E(k) - U(Np) - UB(Np))/(2*t0));
            ka = acos(ck);
            sig2(Np, Np) = -t0*exp(1i*ka);
            gam2 = 1i*(sig2 - sig2');
            % Функция Грина
            G = inv(E(k)*eye(Np) - H - diag(U) - sig1 - sig2);
            TM(k) = real(trace(gam1*G*gam2*G'));
            I = I + (dE*IE*TM(k)*(f1(k) - f2(k)));
        end
        II(iV) = I;
    end
end
