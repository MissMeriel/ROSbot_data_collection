% Take the array from extract_loss_values_pub.py
% graph the loss for a slurmout

mean_loss_array = []; % Insert the array here


% Plotting the graph with x-axis values as factors of five and highlighting the highest and lowest loss values
figure;
ylim([0 0.3])
%xscale=1;
plot(mean_loss_array(:, 1), mean_loss_array(:, 2), '-o');
hold on;

% Highlighting the highest and lowest loss values
[max_loss, max_idx] = max(mean_loss_array(:, 2));
[min_loss, min_idx] = min(mean_loss_array(:, 2));

max_epoch = mean_loss_array(max_idx, 1);
min_epoch = mean_loss_array(min_idx, 1);

scatter(max_epoch, max_loss, 'r', 'filled');
scatter(min_epoch, min_loss, 'g', 'filled');

% Adding labels for the highlighted points
text(max_epoch, max_loss, sprintf('Highest Loss: %.4f (Epoch %d)', max_loss, round(max_epoch/25)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
text(min_epoch, min_loss, sprintf('Lowest Loss: %.4f (Epoch %d)', min_loss, round(min_epoch/25)), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');

% Setting x-axis values as factors of five
%xticks(0:5:max(mean_loss_array(:, 1)));
xticks(5);
%xticks(range(0, int(mean_loss_array[:, 0].max()) + 1, 5))
ylim([0.0, 0.03]);  % Adjust ylim as you see fit

xlabel('Epoch');
ylabel('Loss');
title('Mean Loss per Epoch');
legend('Mean Loss', 'Highest Loss', 'Lowest Loss');
grid on;
hold off;