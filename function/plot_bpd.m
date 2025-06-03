% 假设数据为 SBP 和 DBP
edges = 20:5:220; % 设置直方图的边界

% 绘制 SBP 数据的直方图
histogram(SBP_combined, edges, 'FaceColor', 'r', 'FaceAlpha', 0.7, 'EdgeColor', 'k', 'Normalization', 'probability');
hold on;

% 绘制 DBP 数据的直方图
histogram(DBP_combined, edges, 'FaceColor', 'b', 'FaceAlpha', 0.7, 'EdgeColor', 'k', 'Normalization', 'probability');

% 设置图例
legend({'SBP', 'DBP'}, 'Location', 'northeast');

% 设置标题和轴标签
xlabel('Blood Pressure (mmHg)');
ylabel('Probability');

% 设置图形美化
grid on; % 添加网格线
set(gca, 'FontName', 'Times New Roman','FontSize', 12); % 设置字体大小
xlim([20, 220]); % 限制 x 轴范围
ylim([0, 0.2]); % 限制 y 轴范围

hold off;
