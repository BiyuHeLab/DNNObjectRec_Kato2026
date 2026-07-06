clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir = [rootD '/analysis/Fig3_large_scale_scatter'];
if ~exist(savedir,'dir');mkdir(savedir);end
load([rootD '/rawdata/behav_exp1/summary.mat'])
load([rootD '/rawdata/DNNlabels.mat'])
cnntypes   = {'resnet50','convnextL','cornet-s','vit_l_16',...
    'resnet50-sin','resnet50-blur-st','cornet-s-blur-st',...
    'clip_convnextL_image','clip_vit-l-laion_image',....
    'convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384' 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k'};
Cres = [49 130 189];
Ccon = [59 10 117];
Ccor = [0 178 0];
Cvit = [255 105 180];
Chum = [100 100 100];
col     = [Cres;Ccon;Ccor;Cvit;...
    Cres;Ccor;Cres;...
    Ccon;Cvit;...
    Ccon;Cvit]/255;
mark = {'o' 'o' 'o' 'o'...
    'v' '^' '^'...
    'diamond' 'diamond' 'square' 'square'};

xlims = [30 100;0 100;10 70;0 70;0 35;0 60];
ylims = [0 0.5;0 0.7;0 0.7;0 0.7;0 0.6;0 0.7];
sz = 30;
imtype   = {'Fil' 'Lin' 'SilTex' 'Sil' 'Tex' 'LinFil'};
figlabel = {'Fig.3a' 'Fig.3b' 'Fig.3c' 'Fig.3d' 'Fig.3e' 'Fig.3f'};

figW = 6.5;
figH = 6.5;
[~, colNo]    = ismember(cnntypes,models);
colIdx        = ismember(models,cnntypes);
clipIdx       = contains(models,'clip') &~colIdx;
hum_like_rank = cell(length(models),length(imtype)); % table S2
for type = 1:length(imtype) 
    if ~exist([savedir '/'  imtype{type} '.mat'],'file')
        hum      = squeeze(d_batch_cl(:,type+1,:)); % nImage*nSubject
        nbatch   = size(hum,2);
        hum_accs = squeeze(mean(hum,1,'omitnan')); % avg across image
     
        hum_kappa = nan(nchoosek(nbatch,2),1);
        %-human
        ii = 1;
        for b1 = 1:nbatch
            for b2 = b1+1:nbatch
                hum1 = hum(:,b1);
                hum2 = hum(:,b2);
                excIdx = isnan(hum1)|isnan(hum2);
                hum1(excIdx) = [];
                hum2(excIdx) = [];

                hum1_av = mean(hum1);
                hum2_av = mean(hum2);

                c_exp = hum1_av*hum2_av + (1-hum1_av)*(1-hum2_av);
                c_obs = sum(hum1==hum2)/length(hum1);

                hum_kappa(ii) = (c_obs - c_exp)/(1 - c_exp);
                ii = ii + 1;
            end
        end

        kappa  = nan(nbatch,length(models));
        accs   = nan(1,length(models));
        %-CNN
        for cnn = 1:length(models)
            ann      = get_cnn_resp(rootD,models{cnn},imtype{type});
            if ~strcmp(imtype{type},'Ori')
                ann_base = get_cnn_resp(rootD,models{cnn},'Ori');
                ann(ann_base == 0) = NaN;
            end
            accs(cnn) = mean(ann,'omitnan');
            for bat = 1:nbatch
                hum_b = hum(:,bat);
                ann_b = ann;
                excIdx = isnan(hum_b)|isnan(ann_b);
                hum_b(excIdx) = [];
                ann_b(excIdx) = [];
                hum_av = mean(hum_b);
                ann_av = mean(ann_b);
                c_exp = hum_av*ann_av + (1-hum_av)*(1-ann_av);
                c_obs = sum(hum_b==ann_b)/length(ann_b);
                kappa(bat,cnn) = (c_obs - c_exp)/(1 - c_exp);
            end
        end
        save([savedir '/'  imtype{type} '.mat'],'models','kappa','accs','hum_kappa','hum_accs')
    else
        load([savedir '/'  imtype{type} '.mat'])
    end

    hum_accs_sd  = std(hum_accs*100); 
    hum_kapp_min = min(hum_kappa);
    hum_kapp_max = max(hum_kappa);
    hum_kapp_sd  = std(hum_kappa);
    
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

    scatter(mean(accs(:,~colIdx),1,'omitnan')*100,mean(kappa(:,~colIdx),1,'omitnan'),sz,...
        'MarkerFaceColor',[.5 .5 .5],'MarkerFaceAlpha',0.6,'MarkerEdgeColor','none');
    axis square
    scatter(mean(accs(:,clipIdx),1,'omitnan')*100,mean(kappa(:,clipIdx),1,'omitnan'),sz,...
        'MarkerFaceColor','none','MarkerEdgeColor','r','LineWidth',1);
    for ii = 1:length(colNo)
        scatter(mean(accs(:,colNo(ii)),1,'omitnan')*100,mean(kappa(:,colNo(ii)),1,'omitnan'),sz*2.5,mark{ii},...
            'MarkerFaceColor',col(ii,:),'MarkerFaceAlpha',0.85,'MarkerEdgeColor','none');
    end
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

    [accsZ,accsM,accsS] = zscore(accs);
    [kappZ,kappM,kappS] = zscore(squeeze(mean(kappa,1,'omitnan')));

    hum_accsZ = (mean(hum_accs)/100-accsM)/accsS;
    hum_kappZ = (mean(hum_kappa)-kappM)/kappS;
    accsD = hum_accsZ - accsZ;
    kappD = hum_kappZ - kappZ;

    D = sqrt(accsD.^2 + kappD.^2);

    [~,ord] = sort(D);
    hum_like_rank(:,type) = cellfun(@(x) strrep(x,'BS_',''),models(ord),'UniformOutput',false);
    rectangle('Position',[1 1 figW figH],'EdgeColor','none','FaceColor','none');

    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH])
     
    xlim(xlims(type,:))
    ylim(ylims(type,:))
    xticks(xlims(type,1):10:xlims(type,2))
    yticks(ylims(type,1):.1:ylims(type,2))
    saveas(gcf,[savedir '/' figlabel{type} '.png'])
    exportgraphics(gcf, [savedir '/' imtype{type} '.pdf'], 'ContentType', 'vector');

    % source data
    if ~strcmp(figlabel{type},'')
        t = cell(sum(~colIdx)+1,3);
        t(2:end,1) = cellfun(@(x) strrep(x,'BS_',''),models(~colIdx),'UniformOutput',false);
        t{1,2} = "accuracy (%)";
        t{1,3} = 'error consistency';
        t(2:end,2) = num2cell(mean(accs(:,~colIdx),1,'omitnan')'*100);
        t(2:end,3) = num2cell(mean(kappa(:,~colIdx),1,'omitnan')');
        writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel{type},'WriteVariableNames',false);
    end
end

