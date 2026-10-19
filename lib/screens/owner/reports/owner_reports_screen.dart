import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import '../../../api.dart';
import '../../../theme.dart';
import '../../../widgets/owner_feature_frame.dart';
import '../../../widgets/responsive_shell.dart';

class OwnerReportsScreen extends StatefulWidget {
  final String hostelName;
  final VoidCallback? onBackToDashboard;

  const OwnerReportsScreen({
    super.key,
    required this.hostelName,
    this.onBackToDashboard,
  });

  @override
  State<OwnerReportsScreen> createState() => _OwnerReportsScreenState();
}

class _OwnerReportsScreenState extends State<OwnerReportsScreen> {
  String _selectedRange = 'This Month';

  final List<String> _ranges = [
    'This Week',
    'This Month',
    '3 Months',
    '6 Months',
    'This Year',
  ];

  List<Map<String, dynamic>> _monthlyRevenue = [];
  List<Map<String, dynamic>> _studentGrowth = [];

  bool _isLoadingReports = true;
  String? _reportsError;
  List<Map<String, dynamic>> _complaints = [];
  List<Map<String, dynamic>> _roomPerformance = [];
  List<Map<String, dynamic>> _registeredStudents = [];

  int _totalStudents = 0;
  int _occupiedBeds = 0;
  int _totalBeds = 0;
  int _monthlyRevenueAmount = 0;
  int _feesCollected = 0;
  int _feesPending = 0;
  int _newAdmissions = 0;
  double get _occupancy => _totalBeds == 0 ? 0 : _occupiedBeds / _totalBeds;

  int get _occupancyPercent => (_occupancy * 100).round();
  @override
  void initState() {
    super.initState();
    _loadReports();
  }

  Future<void> _loadReports() async {
    if (mounted) {
      setState(() {
        _isLoadingReports = true;
        _reportsError = null;
      });
    }

    try {
      final response = await Api().getOwnerReports();

      final reportsRaw = response['reports'];

      if (reportsRaw is! List || reportsRaw.isEmpty) {
        throw Exception('No report data found');
      }

      Map<String, dynamic>? report;

      for (final item in reportsRaw) {
        if (item is Map) {
          final data = Map<String, dynamic>.from(item);

          if (data['hostel_name']?.toString().trim().toLowerCase() ==
              widget.hostelName.trim().toLowerCase()) {
            report = data;
            break;
          }
        }
      }

      report ??= Map<String, dynamic>.from(reportsRaw.first as Map);
      final Map<String, dynamic> reportData = report;

      final monthlyRevenue = List<dynamic>.from(
        reportData['monthly_revenue'] ?? [],
      );

      final studentGrowth = List<dynamic>.from(
        reportData['student_growth'] ?? [],
      );

      if (!mounted) return;

      setState(() {
        _monthlyRevenue = monthlyRevenue
            .whereType<Map>()
            .map((e) {
              final row = Map<String, dynamic>.from(e);
              return <String, dynamic>{'month': row['month'] ?? '—', 'value': _asInt(row['value'])};
            })
            .toList();

        _studentGrowth = studentGrowth
            .whereType<Map>()
            .map((e) {
              final row = Map<String, dynamic>.from(e);
              return <String, dynamic>{'month': row['month'] ?? '—', 'value': _asInt(row['value'])};
            })
            .toList();

        final complaintRows = List<dynamic>.from(reportData['recent_complaints'] ?? const [])
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .toList();
        final openComplaints = complaintRows.where((row) => !['resolved', 'closed'].contains(row['status']?.toString().toLowerCase())).length;
        final resolvedComplaints = complaintRows.length - openComplaints;
        _complaints = [
          {'label': 'Open', 'value': openComplaints, 'color': const Color(0xFFC07421)},
          {'label': 'Resolved', 'value': resolvedComplaints, 'color': const Color(0xFF3D7B53)},
        ];
        _roomPerformance = List<dynamic>.from(reportData['room_performance'] ?? const [])
            .whereType<Map>()
            .map((raw) {
              final row = Map<String, dynamic>.from(raw);
              final capacity = _asInt(row['capacity']);
              final occupied = _asInt(row['occupied']);
              final rent = _asInt(row['monthly_rent']);
              return <String, dynamic>{
                'room': row['room_number'] ?? 'Room',
                'occupancy': capacity == 0 ? 0.0 : (occupied / capacity).clamp(0.0, 1.0),
                'revenue': rent * occupied,
              };
            })
            .toList();

        _totalStudents = _asInt(reportData['students']);

        _occupiedBeds = _asInt(reportData['occupied']);

        _totalBeds = _asInt(reportData['total']);

        _monthlyRevenueAmount = _asInt(reportData['revenue']);

        _feesCollected = _asInt(reportData['fees_collected']);
        _feesPending = _asInt(reportData['fees_pending']);
        _registeredStudents = List<dynamic>.from(
          reportData['registered_students'] ?? const [],
        ).whereType<Map>().map((row) => Map<String, dynamic>.from(row)).toList();
        _newAdmissions = 0;

        _isLoadingReports = false;
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _isLoadingReports = false;
        _reportsError = e.toString();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoadingReports) {
      return const ResponsiveShell(
        child: Scaffold(
          backgroundColor: Colors.transparent,
          body: OwnerFeatureFrame(
            child: Center(
              child: CircularProgressIndicator(color: AppTheme.tealMint),
            ),
          ),
        ),
      );
    }

    if (_reportsError != null) {
      return ResponsiveShell(
        child: Scaffold(
          backgroundColor: Colors.transparent,
          body: OwnerFeatureFrame(
            child: Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  'Could not load reports.\n$_reportsError',
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white),
                ),
              ),
            ),
          ),
        ),
      );
    }
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: OwnerFeatureFrame(
          child: CustomScrollView(
            physics: const BouncingScrollPhysics(),
            slivers: [
              SliverToBoxAdapter(child: _buildHeader()),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 18, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildRangeSelector()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 18, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildSummaryGrid()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Occupancy',
                    subtitle: 'Current room and bed utilization',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildOccupancyCard()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Revenue Trend',
                    subtitle: 'Monthly hostel income',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildRevenueChart()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Student Growth',
                    subtitle: 'Admissions over recent months',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildStudentGrowth()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Fee Collection',
                    subtitle: 'Paid and pending hostel fees',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildFeeCollectionCard()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Complaints',
                    subtitle: 'Complaint resolution status',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildComplaintCard()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Room Performance',
                    subtitle: 'Occupancy and revenue by room',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildRoomPerformance()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 40),
                sliver: SliverToBoxAdapter(child: _buildExportSection()),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF073F42), Color(0xFF0A696D), Color(0xFF0A797B)],
        ),
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(32)),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  _circleButton(
                    icon: Icons.arrow_back_rounded,
                    onTap: _goBackToDashboard,
                  ),
                  const Spacer(),
                  _circleButton(
                    icon: Icons.download_outlined,
                    onTap: _showExportSheet,
                  ),
                ],
              ),

              const SizedBox(height: 22),

              Text(
                'Reports & Analytics',
                style: GoogleFonts.playfairDisplay(
                  fontSize: 29,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),

              const SizedBox(height: 5),

              Text(
                widget.hostelName,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  fontSize: 12.5,
                  color: Colors.white.withOpacity(0.65),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _circleButton({required IconData icon, required VoidCallback onTap}) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.13),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white.withOpacity(0.18)),
          ),
          child: Icon(icon, color: Colors.white, size: 20),
        ),
      ),
    );
  }

  void _goBackToDashboard() {
    if (widget.onBackToDashboard != null) {
      widget.onBackToDashboard!();
      return;
    }
    Navigator.of(context).maybePop();
  }

  Widget _buildRangeSelector() {
    return SizedBox(
      height: 40,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _ranges.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final range = _ranges[index];
          final selected = range == _selectedRange;

          return GestureDetector(
            onTap: () {
              setState(() {
                _selectedRange = range;
              });
            },
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: selected ? AppTheme.teal : OwnerFeaturePalette.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: selected ? AppTheme.teal : OwnerFeaturePalette.border,
                ),
              ),
              child: Text(
                range,
                style: GoogleFonts.dmSans(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: selected
                      ? Colors.white
                      : OwnerFeaturePalette.textMuted,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildSummaryGrid() {
    final items = [
      _ReportStat(
        title: 'Students',
        value: '$_totalStudents',
        subtitle: '+$_newAdmissions this month',
        icon: Icons.groups_rounded,
        color: AppTheme.teal,
        background: AppTheme.teal.withOpacity(0.07),
      ),
      _ReportStat(
        title: 'Occupancy',
        value: '$_occupancyPercent%',
        subtitle: '$_occupiedBeds of $_totalBeds beds',
        icon: Icons.bed_rounded,
        color: const Color(0xFF3D7B53),
        background: const Color(0xFFEDF6EF),
      ),
      _ReportStat(
        title: 'Revenue',
        value: 'PKR ${_compactAmount(_monthlyRevenueAmount)}',
        subtitle: 'Based on recorded payments',
        icon: Icons.payments_rounded,
        color: const Color(0xFF3478B7),
        background: const Color(0xFFEAF3FB),
      ),
      _ReportStat(
        title: 'Pending Fees',
        value: 'PKR ${_compactAmount(_feesPending)}',
        subtitle: 'No fee data available',
        icon: Icons.receipt_long_rounded,
        color: const Color(0xFFC07421),
        background: const Color(0xFFFFF4E6),
      ),
    ];

    return LayoutBuilder(
      builder: (context, constraints) {
        final cardWidth = (constraints.maxWidth - 12) / 2;

        return Wrap(
          spacing: 12,
          runSpacing: 12,
          children: items.map((item) {
            return SizedBox(width: cardWidth, child: _reportStatCard(item));
          }).toList(),
        );
      },
    );
  }

  Widget _reportStatCard(_ReportStat item) {
    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: OwnerFeaturePalette.border, width: 0.6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: item.background,
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(item.icon, size: 20, color: item.color),
          ),

          const SizedBox(height: 12),

          Text(
            item.value,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.dmSans(
              fontSize: 17,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),

          const SizedBox(height: 2),

          Text(
            item.title,
            style: GoogleFonts.dmSans(
              fontSize: 10,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),

          const SizedBox(height: 6),

          Text(
            item.subtitle,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.dmSans(
              fontSize: 9,
              fontWeight: FontWeight.w600,
              color: item.color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle({required String title, required String subtitle}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.playfairDisplay(
            fontSize: 19,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          subtitle,
          style: GoogleFonts.dmSans(fontSize: 10.5, color: Colors.white70),
        ),
      ],
    );
  }

  Widget _buildOccupancyCard() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        children: [
          Row(
            children: [
              SizedBox(
                width: 92,
                height: 92,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox(
                      width: 92,
                      height: 92,
                      child: CircularProgressIndicator(
                        value: _occupancy,
                        strokeWidth: 9,
                        backgroundColor: AppTheme.teal.withOpacity(0.08),
                        valueColor: const AlwaysStoppedAnimation<Color>(
                          AppTheme.teal,
                        ),
                      ),
                    ),
                    Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          '$_occupancyPercent%',
                          style: GoogleFonts.dmSans(
                            fontSize: 18,
                            fontWeight: FontWeight.w800,
                            color: OwnerFeaturePalette.text,
                          ),
                        ),
                        Text(
                          'occupied',
                          style: GoogleFonts.dmSans(
                            fontSize: 8.5,
                            color: OwnerFeaturePalette.textMuted,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),

              const SizedBox(width: 18),

              Expanded(
                child: Column(
                  children: [
                    _occupancyRow(
                      label: 'Occupied Beds',
                      value: '$_occupiedBeds',
                      color: AppTheme.teal,
                    ),
                    const SizedBox(height: 12),
                    _occupancyRow(
                      label: 'Available Beds',
                      value: '${_totalBeds - _occupiedBeds}',
                      color: const Color(0xFF3D7B53),
                    ),
                    const SizedBox(height: 12),
                    _occupancyRow(
                      label: 'Total Beds',
                      value: '$_totalBeds',
                      color: OwnerFeaturePalette.textMuted,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _occupancyRow({
    required String label,
    required String value,
    required Color color,
  }) {
    return Row(
      children: [
        Container(
          width: 9,
          height: 9,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            label,
            style: GoogleFonts.dmSans(
              fontSize: 10.5,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),
        ),
        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 12,
            fontWeight: FontWeight.w700,
            color: OwnerFeaturePalette.text,
          ),
        ),
      ],
    );
  }

  Widget _buildRevenueChart() {
    final maxValue = _monthlyRevenue.fold<int>(1, (max, item) => _asInt(item['value']) > max ? _asInt(item['value']) : max);

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 18, 16, 14),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'PKR ${_formatAmount(_monthlyRevenueAmount)}',
                style: GoogleFonts.dmSans(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                  color: OwnerFeaturePalette.text,
                ),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
                decoration: BoxDecoration(
                  color: const Color(0xFFEDF6EF),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.trending_up_rounded,
                      size: 14,
                      color: Color(0xFF3D7B53),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      '8.4%',
                      style: GoogleFonts.dmSans(
                        fontSize: 9.5,
                        fontWeight: FontWeight.w700,
                        color: const Color(0xFF3D7B53),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 22),

          SizedBox(
            height: 170,
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: _monthlyRevenue.map((item) {
                final value = _asInt(item['value']);
                final ratio = value / maxValue;

                return Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        Expanded(
                          child: Align(
                            alignment: Alignment.bottomCenter,
                            child: FractionallySizedBox(
                              heightFactor: ratio.clamp(0.0, 1.0),
                              child: Container(
                                width: double.infinity,
                                constraints: const BoxConstraints(maxWidth: 30),
                                decoration: BoxDecoration(
                                  color: AppTheme.teal,
                                  borderRadius: const BorderRadius.vertical(
                                    top: Radius.circular(8),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          item['month'].toString(),
                          style: GoogleFonts.dmSans(
                            fontSize: 9,
                            color: OwnerFeaturePalette.textMuted,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStudentGrowth() {
    final maxStudents = _studentGrowth.fold<int>(1, (max, item) => _asInt(item['value']) > max ? _asInt(item['value']) : max);

    return Container(
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: _growthInfo(
                  label: 'Current Students',
                  value: '$_totalStudents',
                  color: AppTheme.teal,
                ),
              ),
              Container(
                width: 1,
                height: 42,
                color: OwnerFeaturePalette.border,
              ),
              Expanded(
                child: _growthInfo(
                  label: 'New Admissions',
                  value: '+$_newAdmissions',
                  color: const Color(0xFF3D7B53),
                ),
              ),
            ],
          ),

          const SizedBox(height: 20),

          ..._studentGrowth.map((item) {
            final value = _asInt(item['value']);
            final ratio = value / maxStudents;

            return Padding(
              padding: const EdgeInsets.only(bottom: 11),
              child: Row(
                children: [
                  SizedBox(
                    width: 32,
                    child: Text(
                      item['month'].toString(),
                      style: GoogleFonts.dmSans(
                        fontSize: 9.5,
                        color: OwnerFeaturePalette.textMuted,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: LinearProgressIndicator(
                        value: ratio,
                        minHeight: 8,
                        backgroundColor: AppTheme.teal.withOpacity(0.07),
                        valueColor: const AlwaysStoppedAnimation<Color>(
                          AppTheme.teal,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  SizedBox(
                    width: 25,
                    child: Text(
                      '$value',
                      textAlign: TextAlign.right,
                      style: GoogleFonts.dmSans(
                        fontSize: 10.5,
                        fontWeight: FontWeight.w700,
                        color: OwnerFeaturePalette.text,
                      ),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _growthInfo({
    required String label,
    required String value,
    required Color color,
  }) {
    return Column(
      children: [
        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 19,
            fontWeight: FontWeight.w800,
            color: color,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(
            fontSize: 9.5,
            color: OwnerFeaturePalette.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _buildFeeCollectionCard() {
    final total = _feesCollected + _feesPending;
    final collectedRatio = total == 0 ? 0.0 : _feesCollected / total;

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: _feeInfo(
                  title: 'Collected',
                  amount: _feesCollected,
                  color: const Color(0xFF3D7B53),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _feeInfo(
                  title: 'Pending',
                  amount: _feesPending,
                  color: const Color(0xFFC07421),
                ),
              ),
            ],
          ),

          const SizedBox(height: 20),

          Row(
            children: [
              Text(
                '${(collectedRatio * 100).round()}% collected',
                style: GoogleFonts.dmSans(
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  color: OwnerFeaturePalette.text,
                ),
              ),
              const Spacer(),
              Text(
                'PKR ${_formatAmount(total)} total',
                style: GoogleFonts.dmSans(
                  fontSize: 9.5,
                  color: OwnerFeaturePalette.textMuted,
                ),
              ),
            ],
          ),

          const SizedBox(height: 9),

          ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: LinearProgressIndicator(
              value: collectedRatio,
              minHeight: 10,
              backgroundColor: const Color(0xFFC07421).withOpacity(0.10),
              valueColor: const AlwaysStoppedAnimation<Color>(
                Color(0xFF3D7B53),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _feeInfo({
    required String title,
    required int amount,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: color.withOpacity(0.07),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            title == 'Collected'
                ? Icons.check_circle_outline_rounded
                : Icons.schedule_rounded,
            size: 18,
            color: color,
          ),
          const SizedBox(height: 9),
          Text(
            'PKR ${_formatAmount(amount)}',
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.dmSans(
              fontSize: 14,
              fontWeight: FontWeight.w800,
              color: OwnerFeaturePalette.text,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            title,
            style: GoogleFonts.dmSans(
              fontSize: 9.5,
              color: color,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildComplaintCard() {
    final totalComplaints = _complaints.fold<int>(
      0,
      (sum, item) => sum + _asInt(item['value']),
    );

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                width: 70,
                height: 70,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.06),
                  shape: BoxShape.circle,
                ),
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        '$totalComplaints',
                        style: GoogleFonts.dmSans(
                          fontSize: 18,
                          fontWeight: FontWeight.w800,
                          color: OwnerFeaturePalette.text,
                        ),
                      ),
                      Text(
                        'total',
                        style: GoogleFonts.dmSans(
                          fontSize: 8.5,
                          color: OwnerFeaturePalette.textMuted,
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(width: 16),

              Expanded(
                child: Column(
                  children: _complaints.map((item) {
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 5),
                      child: Row(
                        children: [
                          Container(
                            width: 8,
                            height: 8,
                            decoration: BoxDecoration(
                              color: item['color'] as Color,
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              item['label'].toString(),
                              style: GoogleFonts.dmSans(
                                fontSize: 10.5,
                                color: OwnerFeaturePalette.textMuted,
                              ),
                            ),
                          ),
                          Text(
                            '${item['value']}',
                            style: GoogleFonts.dmSans(
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: OwnerFeaturePalette.text,
                            ),
                          ),
                        ],
                      ),
                    );
                  }).toList(),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildRoomPerformance() {
    return Container(
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        children: List.generate(_roomPerformance.length, (index) {
          final item = _roomPerformance[index];

          return Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(14),
                child: Row(
                  children: [
                    Container(
                      width: 42,
                      height: 42,
                      decoration: BoxDecoration(
                        color: AppTheme.teal.withOpacity(0.07),
                        borderRadius: BorderRadius.circular(13),
                      ),
                      child: Icon(
                        Icons.meeting_room_rounded,
                        color: AppTheme.teal,
                        size: 20,
                      ),
                    ),

                    const SizedBox(width: 12),

                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            item['room'].toString(),
                            style: GoogleFonts.dmSans(
                              fontSize: 12.5,
                              fontWeight: FontWeight.w700,
                              color: OwnerFeaturePalette.text,
                            ),
                          ),
                          const SizedBox(height: 5),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(20),
                            child: LinearProgressIndicator(
                              value: _asDouble(item['occupancy']),
                              minHeight: 6,
                              backgroundColor: AppTheme.teal.withOpacity(0.07),
                              valueColor: const AlwaysStoppedAnimation<Color>(
                                AppTheme.teal,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(width: 12),

                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          '${(_asDouble(item['occupancy']) * 100).round()}%',
                          style: GoogleFonts.dmSans(
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            color: AppTheme.teal,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          'PKR ${_compactAmount(_asInt(item['revenue']))}',
                          style: GoogleFonts.dmSans(
                            fontSize: 9,
                            color: OwnerFeaturePalette.textMuted,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),

              if (index < _roomPerformance.length - 1)
                Divider(
                  height: 1,
                  thickness: 0.6,
                  color: OwnerFeaturePalette.border,
                ),
            ],
          );
        }),
      ),
    );
  }

  Widget _buildExportSection() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xFF103F40),
        borderRadius: BorderRadius.circular(22),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 45,
                height: 45,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(13),
                ),
                child: const Icon(
                  Icons.download_rounded,
                  color: AppTheme.tealMint,
                  size: 21,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Export Reports',
                      style: GoogleFonts.dmSans(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Download analytics for $_selectedRange',
                      style: GoogleFonts.dmSans(
                        fontSize: 10,
                        color: Colors.white.withOpacity(0.55),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          Row(
            children: [
              Expanded(
                child: _exportButton(
                  icon: Icons.picture_as_pdf_outlined,
                  label: 'Download PDF',
                  onTap: () {
                    _exportPdf();
                  },
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _exportButton(
                  icon: Icons.table_chart_outlined,
                  label: 'Excel',
                  onTap: () {
                    _exportAnalyticsCsv();
                  },
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _exportButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return OutlinedButton.icon(
      onPressed: onTap,
      icon: Icon(icon, size: 17),
      label: Text(
        label,
        style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w700),
      ),
      style: OutlinedButton.styleFrom(
        foregroundColor: Colors.white,
        side: BorderSide(color: Colors.white.withOpacity(0.20)),
        minimumSize: const Size.fromHeight(47),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      ),
    );
  }

  Future<void> _exportPdf() async {
    final pdf = pw.Document();
    final currency = (num value) => 'PKR ${value.toStringAsFixed(0)}';
    final reportRows = <List<String>>[
      ['Registered students', '$_totalStudents'],
      ['Occupied beds', '$_occupiedBeds / $_totalBeds'],
      ['Fees collected', currency(_feesCollected)],
      ['Fees pending', currency(_feesPending)],
      ['Open complaints', '${_complaints.length}'],
    ];

    pdf.addPage(
      pw.MultiPage(
        build: (_) => [
          pw.Text(
            'StayBuddy Owner Report',
            style: pw.TextStyle(fontSize: 21, fontWeight: pw.FontWeight.bold),
          ),
          pw.SizedBox(height: 5),
          pw.Text('${widget.hostelName} · $_selectedRange'),
          pw.SizedBox(height: 16),
          pw.Text('Summary', style: pw.TextStyle(fontWeight: pw.FontWeight.bold)),
          pw.SizedBox(height: 6),
          pw.Table.fromTextArray(
            headers: const ['Metric', 'Value'],
            data: reportRows,
          ),
          pw.SizedBox(height: 18),
          pw.Text(
            'Registered Students',
            style: pw.TextStyle(fontWeight: pw.FontWeight.bold),
          ),
          pw.SizedBox(height: 6),
          if (_registeredStudents.isEmpty)
            pw.Text('No registered students found.')
          else
            pw.Table.fromTextArray(
              headers: const ['Student', 'Floor', 'Room', 'Bed', 'Fee status'],
              data: _registeredStudents.map((student) {
                final remaining = _asAmount(student['remaining_fee']);
                return [
                  student['name']?.toString() ?? 'Student',
                  student['floor_label']?.toString() ?? '—',
                  student['room_number']?.toString() ?? '—',
                  student['bed_number']?.toString() ?? '—',
                  remaining <= 0 ? 'Paid' : 'Due ${currency(remaining)}',
                ];
              }).toList(),
            ),
        ],
      ),
    );
    await Printing.sharePdf(
      bytes: await pdf.save(),
      filename: 'staybuddy-owner-report-${DateTime.now().toIso8601String().split('T').first}.pdf',
    );
  }

  Future<void> _exportAnalyticsCsv() async {
    try {
      final csv = await Api().getOwnerExportCsv('analytics');
      await Clipboard.setData(ClipboardData(text: csv));
      _showTemporary('Analytics CSV copied to the clipboard.');
    } catch (error) {
      _showTemporary(error.toString());
    }
  }

  num _asAmount(dynamic value) => num.tryParse('${value ?? 0}') ?? 0;

  int _asInt(dynamic value) => value is num ? value.toInt() : int.tryParse('${value ?? 0}') ?? 0;
  double _asDouble(dynamic value) => value is num ? value.toDouble() : double.tryParse('${value ?? 0}') ?? 0.0;

  void _showExportSheet() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.fromLTRB(18, 12, 18, 18),
            decoration: BoxDecoration(
              color: OwnerFeaturePalette.surfaceStrong,
              borderRadius: BorderRadius.circular(26),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 42,
                  height: 4,
                  margin: const EdgeInsets.only(bottom: 16),
                  decoration: BoxDecoration(
                    color: OwnerFeaturePalette.border,
                    borderRadius: BorderRadius.circular(3),
                  ),
                ),

                _exportMenuTile(
                  icon: Icons.picture_as_pdf_outlined,
                  title: 'Download PDF',
                  subtitle: 'Save the full analytics report',
                  onTap: () {
                    Navigator.pop(context);
                    _exportPdf();
                  },
                ),

                _exportMenuTile(
                  icon: Icons.table_chart_outlined,
                  title: 'Export Excel',
                  subtitle: 'Download structured report data',
                  onTap: () {
                    Navigator.pop(context);
                    _exportAnalyticsCsv();
                  },
                ),

                _exportMenuTile(
                  icon: Icons.share_outlined,
                  title: 'Share Report',
                  subtitle: 'Share analytics summary',
                  onTap: () {
                    Navigator.pop(context);
                    _exportPdf();
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _exportMenuTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return ListTile(
      onTap: onTap,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      leading: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(0.07),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(icon, color: AppTheme.teal, size: 20),
      ),
      title: Text(
        title,
        style: GoogleFonts.dmSans(
          fontSize: 12.5,
          fontWeight: FontWeight.w600,
          color: OwnerFeaturePalette.text,
        ),
      ),
      subtitle: Text(
        subtitle,
        style: GoogleFonts.dmSans(
          fontSize: 9.5,
          color: OwnerFeaturePalette.textMuted,
        ),
      ),
      trailing: const Icon(Icons.chevron_right_rounded),
    );
  }

  String _formatAmount(int amount) {
    final value = amount.toString();
    final buffer = StringBuffer();

    for (int i = 0; i < value.length; i++) {
      final remaining = value.length - i;

      buffer.write(value[i]);

      if (remaining > 1 && remaining % 3 == 1) {
        buffer.write(',');
      }
    }

    return buffer.toString();
  }

  String _compactAmount(int amount) {
    if (amount >= 1000000) {
      return '${(amount / 1000000).toStringAsFixed(1)}M';
    }

    if (amount >= 1000) {
      return '${(amount / 1000).toStringAsFixed(amount % 1000 == 0 ? 0 : 1)}K';
    }

    return amount.toString();
  }

  void _showTemporary(String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: const Color(0xFF123F40),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Text(
            message,
            style: GoogleFonts.dmSans(color: Colors.white),
          ),
        ),
      );
  }
}

class _ReportStat {
  final String title;
  final String value;
  final String subtitle;
  final IconData icon;
  final Color color;
  final Color background;

  const _ReportStat({
    required this.title,
    required this.value,
    required this.subtitle,
    required this.icon,
    required this.color,
    required this.background,
  });
}
