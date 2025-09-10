classdef TestMappedFD2 < matlab.unittest.TestCase
    % Requires on path:
    %   - buildMappedFD.m
    %   - getNonCompactFDmatrix.m

    properties (Constant, Access=private)
        domain    = [0, 1];                 % single domain
        n_list    = 10 .* (2.^(0:7));       % refinements (doubling)
        tolOrder  = 0.7;                    % allowed |p_est - ooa|
        betaFixed = 2.0;                    % FIXED mapping strength across sweep
        trimBC    = true;                   % trim boundary rows when measuring error?
    end

    properties (TestParameter)
        % Accuracy orders to test
        ooa = struct('OOA2',2,'OOA4',4);
        % Derivative orders (supported by buildMappedFD)
        m   = struct('d1',1,'d2',2);
        % Functions to test
        funcName = struct('exp_decay','exp_decay', ...
                          'sine','sine', ...
                          'poly5','poly5');
    end

    methods (TestClassSetup)
        function checkDependencies(tc)
            mustHave = {'buildMappedFD','getNonCompactFDmatrix'};
            for k = 1:numel(mustHave)
                tc.assumeTrue(exist(mustHave{k}, 'file')==2, ...
                    sprintf('Missing required function: %s', mustHave{k}));
            end
        end
    end

    methods (Test)
        function testConvergence(tc, ooa, m, funcName)
            % Arrange
            x1 = tc.domain(1); 
            xn = tc.domain(2);
            args.beta = tc.betaFixed;              % FIXED mapping across all n
            [ffun, dmfcn] = TestMappedFD2.getTestFunction(funcName);

            errs = zeros(numel(tc.n_list),1);
            refs = zeros(numel(tc.n_list),1);

            % boundary trim width (skip one-sided rows in the norm)
            if tc.trimBC
                trim = max(1, ceil(ooa/2));       % conservative trim
            else
                trim = 0;
            end

            % Act (compute errors for refinements)
            for i = 1:numel(tc.n_list)
                n = tc.n_list(i);
                [D, x] = buildMappedFD(x1, xn, n, m, ooa, args);
                f  = ffun(x, x1).';
                dm = dmfcn(x, x1, m).';

                if trim>0 && n>2*trim
                    idx = (trim+1):(n-trim);      % interior only
                    diff_i = D(idx,:)*f - dm(idx);
                    errs(i) = norm(diff_i, inf);
                    refs(i) = norm(dm(idx), inf);
                else
                    diff = D*f - dm;
                    errs(i) = norm(diff, inf);
                    refs(i) = norm(dm,   inf);
                end
            end

            % Pairwise observed orders with general ratio (robust even if not exactly doubling)
            h = 1./(tc.n_list - 1);
            p = log(errs(1:end-1) ./ max(errs(2:end), eps)) ./ log(h(1:end-1) ./ h(2:end));
            p_est = TestMappedFD2.meanOfTail(p, 2);     % use the tail (closer to asymptotic)

            % Assertions
            tc.verifyGreaterThanOrEqual(p_est, ooa - tc.tolOrder, ...
                sprintf('Observed order too low: p_est=%.3f vs ooa=%d', p_est, ooa));
            tc.verifyLessThan(errs(end), errs(1), ...
                'Error at finest grid should be smaller than coarsest.');
            rels = errs ./ max(refs, eps);
            tc.verifyLessThan(isfinite(rels(end)) * rels(end), 1, ...
                'Relative error at finest grid should be < 1.');

            % Diagnostics (shown only on failure)
            tc.addTeardown(@() TestMappedFD2.diagPrint(tc.domain, funcName, m, ooa, ...
                tc.n_list, errs, p, p_est, trim));
        end
    end

    % ---------- helpers ----------
    methods (Static, Access=private)
        function [ffun, dmfcn] = getTestFunction(name)
            switch name
                case 'exp_decay'
                    % gentler decay constant for testing order
                    k = 5;
                    ffun  = @(x,x1) exp(-k*(x - x1));
                    dmfcn = @(x,x1,m) (-k).^m .* exp(-k*(x - x1));
                case 'sine'
                    a = 7; b = 0.3;
                    ffun  = @(x,x1) sin(a*(x - x1) + b);
                    % d^m/dx^m sin(ax+b) = a^m sin(a(x-x1)+b + m*pi/2)
                    dmfcn = @(x,x1,m) (a.^m) .* sin(a*(x - x1) + b + m*pi/2);
                case 'poly5'
                    ffun  = @(x,x1) (x - x1 + 0.1).^5;
                    dmfcn = @(x,x1,m) TestMappedFD2.polyDeriv(x - x1 + 0.1, 5, m);
                otherwise
                    error('Unknown test function.');
            end
        end

        function d = polyDeriv(z, pdeg, m)
            if m > pdeg, d = zeros(size(z)); return; end
            d = prod(pdeg - (0:m-1)) * z.^(pdeg - m);
        end

        function y = meanOfTail(v, k)
            if isempty(v), y = NaN; return; end
            y = mean(v(end-min(k,numel(v))+1:end));
        end

        function diagPrint(domain, funcName, m, ooa, n_list, errs, p, p_est, trim)
            fprintf('\n[Diagnostics] Domain=[%g,%g], f=%s, m=%d, ooa=%d, trim=%d\n', ...
                domain(1), domain(2), funcName, m, ooa, trim);
            T = table(n_list(:), errs(:), 'VariableNames', {'n','abs_err_inf'});
            disp(T);
            fprintf('Pairwise orders: %s\n', sprintf('%.2f ', p));
            fprintf('p_est=%.3f\n\n', p_est);
        end
    end
end
