clear all
hbar=1.06e-34;
q=1.6e-19;
m_t = .25*9.1e-31;
Ef_t = 0.1;
a_t = 3e-10;
m = .250*9.1e-31:.001*9.1e-31:.300*9.1e-31;
Ef = 0.050:0.001:0.110;
a = 2.90e-10:0.01e-10:3.05e-10;
NS=15;NC=16;ND=15;Np=NS+NC+ND;
lev = [0.5];
U_c = [];
U_t = [];
for i=1:length(lev)
    U_c(i,:) = [zeros(NS,1);lev(i)*ones(4,1);zeros(NC-8,1);lev(i)*ones(4,1);zeros(ND,1)];
    U_t(i,:) = [0.1*ones(NS,1);lev(i)*ones(4,1);0.1*ones(NC-8,1);lev(i)*ones(4,1);0.1*ones(ND,1)];
end
l = 0;
in = [m_t*1e30, Ef_t, a_t* 1e9, NS /100, NC / 100, ND / 100, Np / 100];
fd_1 = fopen('C:\Users\nikve\Desktop\Диплом\data_sets\input_p_1.csv','a');
fd_2 = fopen('C:\Users\nikve\Desktop\Диплом\data_sets\output_p_1.csv', 'a');
for i=1:length(m)
    for j=1:length(Ef)
        for k=1:length(a)
            for n=1:length(lev)
                if m(i) == m_t && Ef(j) == Ef_t && a(k) == a_t && n == 3
                else
                    l
                    l = l + 1;
                    t0=(hbar^2)/(2*m(i)*(a(k)^2)*q);
                    UB = U_c(n, :);
                    U = U_t(n, :);
                    T=(2*t0*diag(ones(1,Np)))-(t0*diag(ones(1,Np-1),1))-(t0*diag(ones(1,Np-1),-1));
                    T=T+diag(UB);
                    I = create_I(m(i), Ef(j), a(k), NS, NC, ND, Np, UB, T) * 1e5;
                    input = {[m(i)*1e30], [Ef(j)], [a(k)* 1e9], U};
                    output = {I};
                    fprintf(fd_1,'%f,', input{1:end});
                    fprintf(fd_1, '\n');
                    fprintf(fd_2,'%f,', output{1:end});
                    fprintf(fd_2, '\n');
                end
            end
        end
    end
end
