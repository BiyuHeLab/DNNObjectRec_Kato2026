%% Exp1
clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
datadir = [rootD '/analysis/Fig3_large_scale_scatter'];
savedir = [rootD '/analysis/FigS3_diffusion_model'];
if ~exist(savedir,'dir');mkdir(savedir);end
load([rootD '/rawdata/behav_exp1/summary.mat'])

xlims = [20 100;0 100;0 70;0 70;0 35;0 60];
ylims = [0 0.5;0 0.7;0 0.7;0 0.7;-0.1 0.6;0 0.7];
sz = 30;
imtype   = {'Fil' 'Lin' 'SilTex' 'Sil' 'Tex' 'LinFil'};
figlabel = cellfun(@(x) ['Fig.S3_' x],imtype,'UniformOutput',false);

figW = 6.5;
figH = 6.5;
t = cell(length(imtype),3);
t{1,2} = "accuracy (%)";
t{1,3} = 'error consistency';
for type = 1:length(imtype)
    load([datadir '/'  imtype{type} '.mat'])
    hum_kapp_min = min(hum_kappa);
    hum_kapp_max = max(hum_kappa);
    hum_accs_sd  = std(hum_accs*100); 
    hum_kapp_sd  = std(hum_kappa);

    base = get_cnn_resp(rootD,'diffusion','Ori');
    diffdat  = get_cnn_resp(rootD,'diffusion',imtype{type});
    diffdat(base==0) = NaN;
    diffaccs = mean(diffdat,'omitnan')*100;

    hum = squeeze(d_batch_cl(:,type+1,:));
    nbatch = size(hum,2);
    diffkappa = nan(nbatch,1);
    for bat = 1:nbatch
        hum_b = hum(:,bat);
        ann_b = diffdat;
        excIdx = isnan(hum_b)|isnan(ann_b);
        hum_b(excIdx) = [];
        ann_b(excIdx) = [];

        hum_av = mean(hum_b);
        ann_av = mean(ann_b);

        c_exp = hum_av*ann_av + (1-hum_av)*(1-ann_av);
        c_obs = sum(hum_b==ann_b)/length(ann_b);

        diffkappa(bat) = (c_obs - c_exp)/(1 - c_exp);
    end

    figure
    hold on
    plot([xlims(type,1) xlims(type,2)],[mean(hum_kappa) mean(hum_kappa)],'--','Color',[.5 .5 .5 .5],'LineWidth',2);
    plot([mean(hum_accs) mean(hum_accs)]*100,[ylims(type,1) ylims(type,2)],'--','Color',[.5 .5 .5 .5],'LineWidth',2);
    xregion(mean(hum_accs)*100-hum_accs_sd,mean(hum_accs)*100+hum_accs_sd,'FaceColor',[.8 .8 .8])
    yregion(mean(hum_kappa)-hum_kapp_sd,mean(hum_kappa)+hum_kapp_sd,'FaceColor',[.8 .8 .8])
    plot([xlims(type,1) xlims(type,2)],[hum_kapp_min hum_kapp_min],':','Color',[.5 .5 .5 .5],'LineWidth',1);
    plot([xlims(type,1) xlims(type,2)],[hum_kapp_max hum_kapp_max],':','Color',[.5 .5 .5 .5],'LineWidth',1);
    plot([min(hum_accs)*100 min(hum_accs)*100],[ylims(type,1) ylims(type,2)],':','Color',[.5 .5 .5 .5],'LineWidth',1);
    plot([max(hum_accs)*100 max(hum_accs)*100],[ylims(type,1) ylims(type,2)],':','Color',[.5 .5 .5 .5],'LineWidth',1);

    scatter(mean(accs,1,'omitnan')*100,mean(kappa,1,'omitnan'),sz,...
        'MarkerFaceColor',[.5 .5 .5],'MarkerFaceAlpha',0.6,'MarkerEdgeColor','none');
    axis square
    scatter(diffaccs,mean(diffkappa,1,'omitnan'),sz*2,'Pentagram',...
        'MarkerFaceColor','r','MarkerEdgeColor','none');
    axis square

    box off
    if type == 1
        xlabel('Accuracy (%)')
        ylabel('Consistensy')
    end
    ax = gca;
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.FontName = 'Helvetica';
    ax.FontSize = 6;
    ax.XTickLabelRotation = 0;
    ylabel('Error consistency')
    xlabel('Accuracy (%)')

    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH])

    xlim(xlims(type,:))
    ylim(ylims(type,:))
    xticks(xlims(type,1):10:xlims(type,2))
    yticks(ylims(type,1):.1:ylims(type,2))
    saveas(gcf,[savedir '/' figlabel{type} '.png'])
    exportgraphics(gcf, [savedir '/' imtype{type} '.pdf'], 'ContentType', 'vector');


    t{type+1,1} = imtype{type};
    t(type+1,2) = num2cell(diffaccs);
    t(type+1,3) = num2cell(mean(diffkappa,1,'omitnan'));
end
writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet','Fig.S3','WriteVariableNames',false);

%% Exp2
clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2025';
addpath([rootD '/codes'])
datadir = [rootD '/analysis/Fig4_large_scale_scatter'];
savedir = [rootD '/analysis/FigS3_diffusion_model'];
if ~exist(savedir,'dir');mkdir(savedir);end
load([rootD '/rawdata/behav_exp2/summary.mat'])

xlims = [0 60;0 80];
ylims = [-0.1 0.6;-0.1 0.6];
imtype  = {'Sparse' 'Dense'};
sz = 30;
figlabel = cellfun(@(x) ['Fig.S3_' x],imtype,'UniformOutput',false);

figW = 6.5;
figH = 6.5;
t = cell(length(imtype),3);
t{1,2} = "accuracy (%)";
t{1,3} = 'error consistency';
for type = 1:length(imtype)
    load([datadir '/'  imtype{type} '.mat'])
    hum_kapp_min = min(hum_kappa);
    hum_kapp_max = max(hum_kappa);
    hum_accs_sd  = std(hum_accs*100); 
    hum_kapp_sd  = std(hum_kappa);

    [base,imName_all] = get_cnn_resp(rootD,'diffusion','Sil');
    [diffdat,imName]   = get_cnn_resp(rootD,'diffusion',imtype{type});
    imName_all = cellfun(@(x) x(1:end-6),imName_all,'UniformOutput',false);
    imName     = cellfun(@(x) x(1:end-7),imName,'UniformOutput',false);
    exp2Idx    = ismember(imName_all,imName);
    diffdat(base(exp2Idx)==0) = NaN;
    diffaccs = mean(diffdat,'omitnan')*100;

    hum = squeeze(d_indv_cl(:,type+1,:));
    nbatch = size(hum,2);
    diffkappa = nan(nbatch,1);
    for bat = 1:nbatch
        hum_b = hum(:,bat);
        ann_b = diffdat;
        excIdx = isnan(hum_b)|isnan(ann_b);
        hum_b(excIdx) = [];
        ann_b(excIdx) = [];

        hum_av = mean(hum_b);
        ann_av = mean(ann_b);

        c_exp = hum_av*ann_av + (1-hum_av)*(1-ann_av);
        c_obs = sum(hum_b==ann_b)/length(ann_b);

        diffkappa(bat) = (c_obs - c_exp)/(1 - c_exp);
    end

    figure
    hold on
    plot([xlims(type,1) xlims(type,2)],[mean(hum_kappa) mean(hum_kappa)],'--','Color',[.5 .5 .5 .5],'LineWidth',2);
    plot([mean(hum_accs) mean(hum_accs)]*100,[ylims(type,1) ylims(type,2)],'--','Color',[.5 .5 .5 .5],'LineWidth',2);
    xregion(mean(hum_accs)*100-hum_accs_sd,mean(hum_accs)*100+hum_accs_sd,'FaceColor',[.8 .8 .8])
    yregion(mean(hum_kappa)-hum_kapp_sd,mean(hum_kappa)+hum_kapp_sd,'FaceColor',[.8 .8 .8])
    plot([xlims(type,1) xlims(type,2)],[hum_kapp_min hum_kapp_min],':','Color',[.5 .5 .5 .5],'LineWidth',1);
    plot([xlims(type,1) xlims(type,2)],[hum_kapp_max hum_kapp_max],':','Color',[.5 .5 .5 .5],'LineWidth',1);
    plot([min(hum_accs)*100 min(hum_accs)*100],[ylims(type,1) ylims(type,2)],':','Color',[.5 .5 .5 .5],'LineWidth',1);
    plot([max(hum_accs)*100 max(hum_accs)*100],[ylims(type,1) ylims(type,2)],':','Color',[.5 .5 .5 .5],'LineWidth',1);

    scatter(mean(accs,1,'omitnan')*100,mean(kappa,1,'omitnan'),sz,...
        'MarkerFaceColor',[.5 .5 .5],'MarkerFaceAlpha',0.6,'MarkerEdgeColor','none');
    axis square
    scatter(diffaccs,mean(diffkappa,1,'omitnan'),sz*2,'Pentagram',...
        'MarkerFaceColor','r','MarkerEdgeColor','none');
    axis square

    box off
    if type == 1
        xlabel('Accuracy (%)')
        ylabel('Consistensy')
    end
    ax = gca;
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.FontName = 'Helvetica';
    ax.FontSize = 6;
    ax.XTickLabelRotation = 0;
    ylabel('Error consistency')
    xlabel('Accuracy (%)')

    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH])

    xlim(xlims(type,:))
    ylim(ylims(type,:))
    xticks(xlims(type,1):10:xlims(type,2))
    yticks(ylims(type,1):.1:ylims(type,2))
    saveas(gcf,[savedir '/' figlabel{type} '.png'])
    exportgraphics(gcf, [savedir '/' imtype{type} '.pdf'], 'ContentType', 'vector');


    t{type+1,1} = imtype{type};
    t(type+1,2) = num2cell(diffaccs);
    t(type+1,3) = num2cell(mean(diffkappa,1,'omitnan'));
end
writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet','Fig.S3','WriteVariableNames',false,'Range','A8');