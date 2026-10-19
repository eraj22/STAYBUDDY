import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../api.dart';
import '../../../theme.dart';
import '../../../widgets/owner_feature_frame.dart';
import '../../../widgets/responsive_shell.dart';
import 'add_warden_screen.dart';

class OwnerWardensScreen extends StatefulWidget {
  final String hostelName;
  final VoidCallback? onBackToDashboard;

  const OwnerWardensScreen({
    super.key,
    required this.hostelName,
    this.onBackToDashboard,
  });

  @override
  State<OwnerWardensScreen> createState() => _OwnerWardensScreenState();
}

class _OwnerWardensScreenState extends State<OwnerWardensScreen> {
  final _searchCtrl = TextEditingController();

  bool _loading = true;
  bool _refreshing = false;

  String? _error;

  int? _hostelId;
  int? _changingWardenId;

  List<Map<String, dynamic>> _wardens = [];

  String _selectedFilter = 'All';

  final _filters = const ['All', 'Active', 'Disabled'];

  @override
  void initState() {
    super.initState();

    _searchCtrl.addListener(_refreshSearch);

    _loadData();
  }

  @override
  void dispose() {
    _searchCtrl.removeListener(_refreshSearch);

    _searchCtrl.dispose();

    super.dispose();
  }

  void _refreshSearch() {
    if (mounted) {
      setState(() {});
    }
  }

  // ==========================================================================
  // LOAD OWNER HOSTEL + WARDENS
  // ==========================================================================

  Future<void> _loadData({bool refresh = false}) async {
    if (refresh) {
      setState(() {
        _refreshing = true;
      });
    } else {
      setState(() {
        _loading = true;
      });
    }

    try {
      final api = Api();

      final hostels = await api.getOwnerHostels();

      Map<String, dynamic>? currentHostel;

      for (final item in hostels) {
        final hostel = Map<String, dynamic>.from(item as Map);

        final backendName =
            hostel['name']?.toString().trim().toLowerCase() ?? '';

        final currentName = widget.hostelName.trim().toLowerCase();

        if (backendName == currentName) {
          currentHostel = hostel;
          break;
        }
      }

      // If owner currently has only one hostel,
      // safely use it even if display formatting
      // caused the name not to match.
      if (currentHostel == null && hostels.length == 1) {
        currentHostel = Map<String, dynamic>.from(hostels.first as Map);
      }

      if (currentHostel == null) {
        throw Exception(
          'Could not determine the current hostel. Please reopen the hostel dashboard.',
        );
      }

      final hostelId = int.tryParse(currentHostel['id'].toString());

      if (hostelId == null) {
        throw Exception('Invalid hostel ID returned by server.');
      }

      final rawWardens = await api.getOwnerWardens();

      final allWardens = rawWardens
          .map((item) => Map<String, dynamic>.from(item as Map))
          .toList();

      final hostelWardens = allWardens
          .where((warden) {
            return int.tryParse(warden['hostel_id'].toString()) == hostelId;
          })
          .map((warden) {
            final permissions = warden['permissions'];

            int permissionCount = 0;

            if (permissions is Map) {
              permissionCount = permissions.values
                  .where((value) => value == true)
                  .length;
            }

            return {
              ...warden,

              'status': 'Offline',

              'accountStatus': warden['account_status'] ?? 'Active',

              'permissionsCount': permissionCount,

              'lastActive': 'Not available',
            };
          })
          .toList();

      if (!mounted) return;

      setState(() {
        _hostelId = hostelId;

        _wardens = hostelWardens;

        _error = null;
      });
    } catch (error) {
      if (!mounted) return;

      setState(() {
        _error = error.toString().replaceFirst('Exception: ', '');
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
          _refreshing = false;
        });
      }
    }
  }

  List<Map<String, dynamic>> get _filteredWardens {
    final query = _searchCtrl.text.trim().toLowerCase();

    return _wardens.where((warden) {
      final name = warden['name']?.toString().toLowerCase() ?? '';

      final email = warden['email']?.toString().toLowerCase() ?? '';

      final phone = warden['phone']?.toString().toLowerCase() ?? '';

      final accountStatus = warden['accountStatus']?.toString();

      final matchesSearch =
          query.isEmpty ||
          name.contains(query) ||
          email.contains(query) ||
          phone.contains(query);

      bool matchesFilter = true;

      if (_selectedFilter == 'Active') {
        matchesFilter = accountStatus == 'Active';
      }

      if (_selectedFilter == 'Disabled') {
        matchesFilter = accountStatus == 'Disabled';
      }

      return matchesSearch && matchesFilter;
    }).toList();
  }

  int get _active =>
      _wardens.where((warden) => warden['accountStatus'] == 'Active').length;

  int get _disabled =>
      _wardens.where((warden) => warden['accountStatus'] == 'Disabled').length;

  // ==========================================================================
  // OPEN ADD WARDEN
  // ==========================================================================

  Future<void> _openAddWarden() async {
    if (_hostelId == null) {
      _showMessage('Hostel information is still loading.', error: true);
      return;
    }

    final result = await Navigator.push<Map<String, dynamic>>(
      context,
      MaterialPageRoute(
        builder: (_) => AddWardenScreen(
          hostelId: _hostelId!,
          hostelName: widget.hostelName,
        ),
      ),
    );

    if (result == null) {
      return;
    }

    await _loadData(refresh: true);

    if (!mounted) return;

    _showMessage('${result['name']} added successfully.');
  }

  Future<void> _changeAccountStatus(Map<String, dynamic> warden) async {
    final wardenId = int.tryParse(warden['id']?.toString() ?? '');
    if (wardenId == null || _changingWardenId != null) return;
    final isActive = warden['accountStatus']?.toString() == 'Active';
    final action = isActive ? 'Disable' : 'Enable';
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text('$action warden?'),
        content: Text(
          isActive
              ? '${warden['name'] ?? 'This warden'} will be signed out and cannot log in until you enable the account again.'
              : '${warden['name'] ?? 'This warden'} will be able to log in and manage this hostel again.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: isActive ? const Color(0xFFB74848) : AppTheme.teal,
            ),
            onPressed: () => Navigator.pop(dialogContext, true),
            child: Text(action),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    setState(() => _changingWardenId = wardenId);
    try {
      final result = await Api().updateWardenAccountStatus(
        wardenId: wardenId,
        enabled: !isActive,
      );
      if (!mounted) return;
      final accountStatus = result['account_status']?.toString() ??
          (isActive ? 'Disabled' : 'Active');
      setState(() {
        _wardens = _wardens.map((item) {
          if (item['id']?.toString() != wardenId.toString()) return item;
          return {...item, 'accountStatus': accountStatus};
        }).toList();
      });
      _showMessage(
        accountStatus == 'Disabled'
            ? 'Warden disabled. Their access has been stopped.'
            : 'Warden enabled successfully.',
      );
    } catch (error) {
      if (mounted) {
        _showMessage(
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
      }
    } finally {
      if (mounted) setState(() => _changingWardenId = null);
    }
  }

  // ==========================================================================
  // UI
  // ==========================================================================

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,

        floatingActionButton: FloatingActionButton.extended(
          onPressed: _loading ? null : _openAddWarden,

          backgroundColor: AppTheme.teal,

          foregroundColor: Colors.white,

          elevation: 3,

          icon: const Icon(Icons.person_add_alt_1_rounded),

          label: Text(
            'Add Warden',
            style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
          ),
        ),

        body: OwnerFeatureFrame(
          child: RefreshIndicator(
            color: AppTheme.teal,
            onRefresh: () => _loadData(refresh: true),
            child: CustomScrollView(
              physics: const AlwaysScrollableScrollPhysics(
                parent: BouncingScrollPhysics(),
              ),
              slivers: [
                SliverToBoxAdapter(child: _header()),

                if (_loading)
                  const SliverFillRemaining(
                    hasScrollBody: false,
                    child: Center(
                      child: CircularProgressIndicator(color: AppTheme.teal),
                    ),
                  )
                else if (_error != null)
                  SliverFillRemaining(
                    hasScrollBody: false,
                    child: _errorState(),
                  )
                else ...[
                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(18, 20, 18, 0),
                    sliver: SliverToBoxAdapter(child: _summary()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(18, 20, 18, 0),
                    sliver: SliverToBoxAdapter(child: _search()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(18, 12, 18, 0),
                    sliver: SliverToBoxAdapter(child: _filtersRow()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                    sliver: SliverToBoxAdapter(
                      child: Row(
                        children: [
                          Text(
                            'WARDEN ACCOUNTS',
                            style: GoogleFonts.dmSans(
                              fontSize: 10.5,
                              letterSpacing: 1.2,
                              fontWeight: FontWeight.w700,
                              color: AppTheme.textMuted,
                            ),
                          ),

                          const Spacer(),

                          Text(
                            '${_filteredWardens.length}',
                            style: GoogleFonts.dmSans(
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: AppTheme.teal,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  if (_filteredWardens.isEmpty)
                    SliverToBoxAdapter(child: _emptyState())
                  else
                    SliverPadding(
                      padding: const EdgeInsets.fromLTRB(18, 0, 18, 110),
                      sliver: SliverList(
                        delegate: SliverChildBuilderDelegate((context, index) {
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 12),
                            child: _wardenCard(_filteredWardens[index]),
                          );
                        }, childCount: _filteredWardens.length),
                      ),
                    ),
                ],
              ],
            ),
          ),
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

  Widget _header() {
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
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 25),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  InkWell(
                    onTap: _goBackToDashboard,
                    borderRadius: BorderRadius.circular(50),
                    child: Container(
                      width: 42,
                      height: 42,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.10),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.arrow_back_rounded,
                        color: Colors.white,
                      ),
                    ),
                  ),

                  const Spacer(),

                  if (_refreshing)
                    const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    ),
                ],
              ),

              const SizedBox(height: 22),

              Text(
                'Wardens',
                style: GoogleFonts.playfairDisplay(
                  fontSize: 30,
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
                  color: Colors.white.withOpacity(0.68),
                ),
              ),

              const SizedBox(height: 15),

              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 11,
                  vertical: 7,
                ),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(30),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.shield_outlined,
                      color: Colors.white,
                      size: 14,
                    ),

                    const SizedBox(width: 6),

                    Text(
                      'Hostel-scoped access',
                      style: GoogleFonts.dmSans(
                        fontSize: 10.5,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _summary() {
    return Row(
      children: [
        Expanded(
          child: _summaryCard(
            title: 'Total',
            value: '${_wardens.length}',
            icon: Icons.groups_rounded,
          ),
        ),

        const SizedBox(width: 10),

        Expanded(
          child: _summaryCard(
            title: 'Active',
            value: '$_active',
            icon: Icons.verified_user_rounded,
          ),
        ),

        const SizedBox(width: 10),

        Expanded(
          child: _summaryCard(
            title: 'Disabled',
            value: '$_disabled',
            icon: Icons.person_off_rounded,
          ),
        ),
      ],
    );
  }

  Widget _summaryCard({
    required String title,
    required String value,
    required IconData icon,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppTheme.tealMint, size: 20),

          const SizedBox(height: 12),

          Text(
            value,
            style: GoogleFonts.playfairDisplay(
              fontSize: 23,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),

          Text(
            title,
            style: GoogleFonts.dmSans(
              fontSize: 10.5,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),
        ],
      ),
    );
  }

  Widget _search() {
    return TextField(
      controller: _searchCtrl,
      decoration: InputDecoration(
        hintText: 'Search name, email or phone',
        hintStyle: GoogleFonts.dmSans(
          color: OwnerFeaturePalette.textMuted,
          fontSize: 12,
        ),
        prefixIcon: const Icon(
          Icons.search_rounded,
          color: OwnerFeaturePalette.textMuted,
        ),
        suffixIcon: _searchCtrl.text.isNotEmpty
            ? IconButton(
                onPressed: () {
                  _searchCtrl.clear();
                },
                icon: const Icon(Icons.close_rounded),
              )
            : null,
        filled: true,
        fillColor: OwnerFeaturePalette.surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: OwnerFeaturePalette.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: OwnerFeaturePalette.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppTheme.teal),
        ),
      ),
    );
  }

  Widget _filtersRow() {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: _filters.map((filter) {
          final selected = _selectedFilter == filter;

          return Padding(
            padding: const EdgeInsets.only(right: 8),
            child: ChoiceChip(
              label: Text(filter),
              selected: selected,
              onSelected: (_) {
                setState(() {
                  _selectedFilter = filter;
                });
              },
              selectedColor: AppTheme.teal.withOpacity(0.12),
              side: BorderSide(
                color: selected ? AppTheme.teal : AppTheme.border,
              ),
              labelStyle: GoogleFonts.dmSans(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: selected ? AppTheme.teal : AppTheme.textPrimary,
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _wardenCard(Map<String, dynamic> warden) {
    final name = warden['name']?.toString() ?? 'Warden';

    final email = warden['email']?.toString() ?? '—';

    final phone = warden['phone']?.toString() ?? '—';

    final accountStatus = warden['accountStatus']?.toString() ?? 'Active';

    final active = accountStatus == 'Active';
    final wardenId = int.tryParse(warden['id']?.toString() ?? '');
    final changing = _changingWardenId != null && _changingWardenId == wardenId;

    String initials = name.isNotEmpty ? name[0].toUpperCase() : 'W';

    final parts = name.trim().split(RegExp(r'\s+'));

    if (parts.length > 1) {
      initials = '${parts.first[0]}${parts.last[0]}'.toUpperCase();
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFF0A696D), Color(0xFF129193)],
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Center(
                  child: Text(
                    initials,
                    style: GoogleFonts.dmSans(
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                ),
              ),

              const SizedBox(width: 13),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            name,
                            style: GoogleFonts.dmSans(
                              fontSize: 13.5,
                              fontWeight: FontWeight.w700,
                              color: OwnerFeaturePalette.text,
                            ),
                          ),
                        ),

                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: active
                                ? const Color(0xFFE8F5EE)
                                : const Color(0xFFF8EAEA),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            accountStatus,
                            style: GoogleFonts.dmSans(
                              fontSize: 9.5,
                              fontWeight: FontWeight.w700,
                              color: active
                                  ? const Color(0xFF27754D)
                                  : const Color(0xFF9B3B3B),
                            ),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 5),

                    Text(
                      email,
                      style: GoogleFonts.dmSans(
                        fontSize: 10.5,
                        color: OwnerFeaturePalette.textMuted,
                      ),
                    ),

                    const SizedBox(height: 2),

                    Text(
                      phone,
                      style: GoogleFonts.dmSans(
                        fontSize: 10.5,
                        color: OwnerFeaturePalette.textMuted,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            child: active
                ? OutlinedButton.icon(
                    onPressed: changing ? null : () => _changeAccountStatus(warden),
                    icon: changing
                        ? const SizedBox(
                            width: 15,
                            height: 15,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.person_off_outlined),
                    label: const Text('Disable Warden'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: const Color(0xFFB74848),
                      side: const BorderSide(color: Color(0xFFB74848)),
                    ),
                  )
                : FilledButton.icon(
                    onPressed: changing ? null : () => _changeAccountStatus(warden),
                    icon: changing
                        ? const SizedBox(
                            width: 15,
                            height: 15,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Icon(Icons.person_add_alt_1_rounded),
                    label: const Text('Enable Warden'),
                    style: FilledButton.styleFrom(
                      backgroundColor: AppTheme.teal,
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _smallDetail(IconData icon, String value) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 15, color: OwnerFeaturePalette.textMuted),

        const SizedBox(width: 5),

        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 10.5,
            fontWeight: FontWeight.w600,
            color: OwnerFeaturePalette.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _emptyState() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(18, 35, 18, 140),
      child: Center(
        child: Column(
          children: [
            Container(
              width: 72,
              height: 72,
              decoration: BoxDecoration(
                color: AppTheme.teal.withOpacity(0.07),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.supervisor_account_outlined,
                color: AppTheme.teal,
                size: 31,
              ),
            ),

            const SizedBox(height: 15),

            Text(
              _searchCtrl.text.trim().isEmpty ? 'No Wardens Yet' : 'No Results',
              style: GoogleFonts.playfairDisplay(
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: OwnerFeaturePalette.text,
              ),
            ),

            const SizedBox(height: 6),

            Text(
              _searchCtrl.text.trim().isEmpty
                  ? 'Create the first Warden account for this hostel.'
                  : 'Try another name, email or phone number.',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                fontSize: 11.5,
                color: OwnerFeaturePalette.textMuted,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _errorState() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(
            Icons.error_outline_rounded,
            size: 45,
            color: Colors.redAccent,
          ),

          const SizedBox(height: 14),

          Text(
            'Unable to load Wardens',
            style: GoogleFonts.playfairDisplay(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),

          const SizedBox(height: 8),

          Text(
            _error ?? '',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              fontSize: 11,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),

          const SizedBox(height: 18),

          ElevatedButton.icon(
            onPressed: _loadData,
            icon: const Icon(Icons.refresh_rounded),
            label: const Text('Try Again'),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.teal,
              foregroundColor: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  void _showMessage(String message, {bool error = false}) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: error
              ? const Color(0xFF963C3C)
              : const Color(0xFF123F40),
          content: Text(
            message,
            style: GoogleFonts.dmSans(color: Colors.white),
          ),
        ),
      );
  }
}
