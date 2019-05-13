%%
t = readtable('IMDB-Movie-Data.csv');
x = t{2:1000,8:11};
y = t{2:1000,12};
m = length(x);
%%

[X, mu, sigma] = featureNormalize(x);
X = [ones(m,1) X];
Y = y;
iter = 1000;
alpha = 0.01;
theta = zeros(5,1);

%%
[m,n] = size(X) ;
P = 0.80 ;
idx = randperm(m)  ;


Training_X = X(idx(1:round(P*m)),:) ; 
Testing_X = X(idx(round(P*m)+1:end),:) ;

Training_Y = Y(idx(1:round(P*m)),:) ; 
Testing_Y = Y(idx(round(P*m)+1:end),:) ;




[theta, J_history] = gradientDescentMulti(Training_X, Training_Y, theta, alpha, iter);
%%
result = Testing_X*theta;

rmse = sqrt(mean(Testing_Y - result).^2);

disp(result);

disp("=====");
disp(rmse);


%%
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


disp("22");

