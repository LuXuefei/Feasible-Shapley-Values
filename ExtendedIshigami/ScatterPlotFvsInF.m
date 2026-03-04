function ScatterPlotFvsInF(x0,x1,X,mypdf,feasi_threshold,plotindx,plotindy)

%% scatter plot of feasible and infeasible points
%figure;
CO = get(gca,"ColorOrder");
scatter(X(mypdf(X) >= feasi_threshold, plotindx), X(mypdf(X) >= feasi_threshold, plotindy),  'filled','Color',CO(1,:),'MarkerFaceAlpha',0.5); hold on;
scatter(X(mypdf(X) < feasi_threshold, plotindx), X(mypdf(X) < feasi_threshold, plotindy), 'filled','Color', CO(3,:),'MarkerFaceAlpha',0.5);
scatter(x0(plotindx), x0(plotindy), 100, 'b', 'filled', 'MarkerEdgeColor','k','LineWidth',1.5);
scatter(x1(plotindx), x1(plotindy), 300, CO(7,:), 'filled', 'pentagram','MarkerEdgeColor','k','LineWidth',1.5);
xlabel(['X',num2str(plotindx)]);
ylabel(['X',num2str(plotindy)]);
ylim([-4,4])
xlim([-4,4])
%title('Scatter Plot of Feasible and Infeasible Points');
legend({'Feasible', 'Infeasible','Baseline x0','Counterfactual x1'}, 'Location', 'best');
% add an arrow from baseline to counterfactual
ax = gca; f = gcf;
xdata = [x0(plotindx) x1(plotindx)];
ydata = [x0(plotindy) x1(plotindy)];
oldAxUnits = ax.Units; oldFigUnits = f.Units;
ax.Units = 'normalized'; f.Units = 'normalized';
axPos = ax.Position;              % [left bottom width height]
pxlim = ax.XLim; pylim = ax.YLim;
xfig = axPos(1) + (xdata - pxlim(1)) ./ diff(pxlim) .* axPos(3);
yfig = axPos(2) + (ydata - pylim(1)) ./ diff(pylim) .* axPos(4);
ax.Units = oldAxUnits; f.Units = oldFigUnits;
annotation('arrow', xfig, yfig, 'Color', 'k', 'LineWidth', 1.5);
set(gca,"FontSize",15)
hold off;