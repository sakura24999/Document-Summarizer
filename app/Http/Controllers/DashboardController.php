<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Document;
use App\Models\DocumentSummary;
use Illuminate\Support\Facades\DB;
use Carbon\Carbon;

class DashboardController extends Controller
{
    /**
     * ダッシュボード画面を表示する
     *
     * @return \Illuminate\View\View
     */
    public function index()
    {
        // ドキュメント総数
        $documentsCount = Document::count();

        // 要約総数
        $summariesCount = DocumentSummary::count();

        // 最近の活動数（過去7日間の文書と要約の合計）
        $recentActivityCount = Document::where('created_at', '>=', Carbon::now()->subDays(7))
            ->count()
            + DocumentSummary::where('created_at', '>=', Carbon::now()->subDays(7))
                ->count();

        // 最近アップロードされた文書（最新5件）
        $recentDocuments = Document::with('summary')
            ->orderBy('created_at', 'desc')
            ->take(5)
            ->get();

        // 文書タイプ別の統計
        $documentsByType = Document::select('file_type', DB::raw('count(*) as count'))
            ->groupBy('file_type')
            ->get()
            ->pluck('count', 'file_type')
            ->toArray();

        // 月別のアップロード統計（過去6ヶ月）
        $monthlyStats = Document::select(
            DB::raw('YEAR(created_at) as year'),
            DB::raw('MONTH(created_at) as month'),
            DB::raw('count(*) as count')
        )
            ->where('created_at', '>=', Carbon::now()->subMonths(6))
            ->groupBy('year', 'month')
            ->orderBy('year')
            ->orderBy('month')
            ->get();

        // チャート用データの整形
        $chartData = [
            'labels' => [],
            'datasets' => [
                [
                    'label' => 'アップロード数',
                    'data' => [],
                    'backgroundColor' => 'rgba(74, 108, 247, 0.5)',
                    'borderColor' => 'rgba(74, 108, 247, 1)',
                ]
            ]
        ];

        // 過去6ヶ月分のラベルとデータを準備
        for ($i = 5; $i >= 0; $i--) {
            $date = Carbon::now()->subMonths($i);
            $year = $date->year;
            $month = $date->month;

            $chartData['labels'][] = $date->format('Y年n月');

            // この月のデータを検索
            $found = false;
            foreach ($monthlyStats as $stat) {
                if ($stat->year == $year && $stat->month == $month) {
                    $chartData['datasets'][0]['data'][] = $stat->count;
                    $found = true;
                    break;
                }
            }

            // データがなければ0をセット
            if (!$found) {
                $chartData['datasets'][0]['data'][] = 0;
            }
        }

        // ビューに変数を渡す
        return view('dashboard', compact(
            'documentsCount',
            'summariesCount',
            'recentActivityCount',
            'recentDocuments',
            'documentsByType',
            'chartData'
        ));
    }

    /**
     * アクティビティログを表示する
     *
     * @return \Illuminate\View\View
     */
    public function activity()
    {
        // 最近の活動をロードする（文書のアップロード、要約の生成など）
        $activities = DB::table('documents')
            ->select(
                'id',
                'title',
                DB::raw("'document_uploaded' as activity_type"),
                'created_at'
            )
            ->unionAll(
                DB::table('document_summaries')
                    ->select(
                        'document_id as id',
                        DB::raw('(SELECT title FROM documents WHERE id = document_summaries.document_id) as title'),
                        DB::raw("'summary_generated' as activity_type"),
                        'created_at'
                    )
            )
            ->orderBy('created_at', 'desc')
            ->take(20)
            ->get();

        return view('activity', compact('activities'));
    }

    /**
     * システム統計情報を表示する
     *
     * @return \Illuminate\View\View
     */
    public function stats()
    {
        // 総文書数
        $totalDocuments = Document::count();

        // 総要約数
        $totalSummaries = DocumentSummary::count();

        // 平均処理時間
        $avgProcessingTime = DocumentSummary::avg('processing_time');

        // 文書タイプ別の統計
        $documentsByType = Document::select('file_type', DB::raw('count(*) as count'))
            ->groupBy('file_type')
            ->get();

        // 月別統計（過去12ヶ月）
        $monthlyStats = Document::select(
            DB::raw('YEAR(created_at) as year'),
            DB::raw('MONTH(created_at) as month'),
            DB::raw('count(*) as document_count')
        )
            ->where('created_at', '>=', Carbon::now()->subMonths(12))
            ->groupBy('year', 'month')
            ->get();

        return view('stats', compact(
            'totalDocuments',
            'totalSummaries',
            'avgProcessingTime',
            'documentsByType',
            'monthlyStats'
        ));
    }
}
