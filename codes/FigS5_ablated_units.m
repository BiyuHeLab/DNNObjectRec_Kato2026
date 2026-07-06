clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir   = [rootD '/analysis/FigS5_ablated_units'];
if ~exist(savedir,'dir');mkdir(savedir);end

imtype    = {'Sparse' 'Dense'};
cnntypes   = {'resnet50','convnextL','cornet-s','vit_l_16',...
    '','resnet50-blur-st','cornet-s-blur-st','resnet50-sin',...
    '','convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384' 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k'};
cnnLabel  = {'inResNet','inConvNeXt','inCORnet','inViT',...
    '','blResNet','blCORnet','stResNet',...
    '','ftConvNeXt' 'ftViT'};
figlabel = 'Fig.S5f';

Cres = [49 130 189];
Ccon = [59 10 117];
Ccor = [0 178 0];
Cvit = [255 105 180];
Chum = [100 100 100];
facec     = [Cres;Ccon;Ccor;Cvit;...
             Cres;Ccor;Cres;...
             Ccon;Cvit];
figW = 5;
figH = 3.85;
frac = nan(2,length(cnntypes));
for type = 1:length(imtype)
    for cnn = 1:length(cnntypes)
        cnntype   = cnntypes{cnn};
        if ~isempty(cnntype)
            units = readcell([savedir '/ablatedUnits_' cnntype '_' imtype{type} '.csv']);
            imgs  = dir([rootD '/rawdata/act/' cnntype '_Ori/*o']);
            actD  = sprintf([rootD '/rawdata/act/%s_Ori'],cnntype);
            load([actD '/' imgs(1).name '/' units{1,1} '.mat']);
            frac(type,cnn) = size(units,1)*100/length(actpat);
        end
    end
end
t = cell(length(cnnLabel(~strcmp(cnnLabel,'')))+1,3);
t(2:end,1) = cnnLabel(~strcmp(cnnLabel,''));
t{1,2} = 'Sparse';
t{1,3} = 'Dense';
t(2:end,2) = num2cell(frac(1,~strcmp(cnnLabel,'')));
t(2:end,3) = num2cell(frac(2,~strcmp(cnnLabel,'')));
writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel,'WriteVariableNames',false);


for type = 1:length(imtype) %
    dat = frac(type,:);
    figure
    b = bar(dat,'FaceColor','flat','LineWidth',1);
    j = 1;
    for ii = 1:length(dat)
        if ~isnan(dat(ii))
            b.CData(ii,:) = facec(j,:)/255;
            j = j + 1;
        end
    end

    ylim([0 100])
    xlim([0 length(cnnLabel)+1])
    ylabel('Silenced unit (%)')
    box off
    xticks(find(~strcmp(cnnLabel,'')))
    xticklabels(strrep(cnnLabel(~strcmp(cnnLabel,'')),'_','\_'))
    ax = gca;
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.FontName = 'Helvetica';
    ax.FontSize = 6;
    rectangle('Position',[1 1 figW figH],'EdgeColor','none','FaceColor','none');
    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH])
    saveas(gcf,[savedir '/' figlabel '_' imtype{type} '.png'])
    exportgraphics(gcf, [savedir '/' imtype{type} '_silencedUn.pdf'], 'ContentType', 'vector');
end