import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import '../widgets/three_d_container.dart';
import 'hostel_detail_screen.dart';
import 'ai_recommendation_screen.dart';
import 'ai_room_search_screen.dart';

class HostelsScreen extends StatefulWidget {
  const HostelsScreen({super.key});

  @override
  State<HostelsScreen> createState() => _HostelsScreenState();
}

class _HostelsScreenState extends State<HostelsScreen> {
  final api = Api();
  bool loading = true;
  List<dynamic> hostels = [];
  String query = '';
  String activeFilter = 'All';

  final List<String> _filters = ['All', 'Available Now', 'Top Rated', 'Nearby'];
  final TextEditingController _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => loading = true);
    try {
      final data = await api.getHostels();
      setState(() {
        hostels = data;
        loading = false;
      });
    } catch (e) {
      setState(() => loading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('$e'),
            backgroundColor: AppTheme.teal,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        );
      }
    }
  }

  List<dynamic> get _filtered {
    final q = query.toLowerCase().trim();
    return hostels.where((h) {
      final name = (h['name'] ?? '').toString().toLowerCase();
      final city = (h['city'] ?? '').toString().toLowerCase();
      final avail = (h['available_capacity'] ?? 0) as int;
      final matchQuery = q.isEmpty || name.contains(q) || city.contains(q);
      final matchFilter = activeFilter == 'All' ||
          (activeFilter == 'Available Now' && avail > 0);
      return matchQuery && matchFilter;
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.40,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: Column(
          children: [
            // Teal header with search
            _buildHeader(),

            // Filter chips
            _buildFilterRow(),

            // AI Room Search Banner
            _buildAiBanner(),

            // Results count
            if (!loading)
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 10, 20, 4),
                child: Row(
                  children: [
                    Text(
                      '${_filtered.length} hostels found',
                      style: GoogleFonts.dmSans(
                        fontSize: 12,
                        color: AppTheme.textMuted,
                      ),
                    ),
                  ],
                ),
              ),

            // List
            Expanded(
              child: loading
                  ? Center(
                      child: CircularProgressIndicator(
                        color: AppTheme.teal,
                        strokeWidth: 2,
                      ),
                    )
                  : _filtered.isEmpty
                      ? _buildEmpty()
                      : RefreshIndicator(
                          color: AppTheme.teal,
                          onRefresh: _load,
                          child: ListView.builder(
                            padding: const EdgeInsets.fromLTRB(16, 4, 16, 20),
                            itemCount: _filtered.length,
                            itemBuilder: (context, i) =>
                                _HostelCard(
                                  hostel: _filtered[i],
                                  index: i,
                                  onTap: () {
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (_) => HostelDetailScreen(
                                          id: _filtered[i]['id'],
                                        ),
                                      ),
                                    );
                                  },
                                ),
                          ),
                        ),
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
          colors: [AppTheme.tealDeep, AppTheme.teal],
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '📍 Islamabad, Pakistan',
                          style: GoogleFonts.dmSans(
                            color: Colors.white.withOpacity(0.65),
                            fontSize: 12,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          'Discover Hostels',
                          style: GoogleFonts.playfairDisplay(
                            fontSize: 22,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  ),
                  // Avatar/profile button
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.15),
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: Colors.white.withOpacity(0.3),
                      ),
                    ),
                    child: const Icon(
                      Icons.person_rounded,
                      size: 22,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              // Search bar
              Container(
                height: 46,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(
                    color: Colors.white.withOpacity(0.2),
                  ),
                ),
                child: TextField(
                  controller: _searchController,
                  onChanged: (v) => setState(() => query = v),
                  style: GoogleFonts.dmSans(color: Colors.white, fontSize: 14),
                  decoration: InputDecoration(
                    hintText: 'Search hostel or city...',
                    hintStyle: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.55),
                      fontSize: 14,
                    ),
                    prefixIcon: const Icon(
                      Icons.search_rounded,
                      color: Colors.white70,
                      size: 20,
                    ),
                    suffixIcon: query.isNotEmpty
                        ? IconButton(
                            icon: const Icon(
                              Icons.close_rounded,
                              color: Colors.white70,
                              size: 18,
                            ),
                            onPressed: () {
                              _searchController.clear();
                              setState(() => query = '');
                            },
                          )
                        : null,
                    border: InputBorder.none,
                    contentPadding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              // ── AI quick-access buttons ──────────────────────────
              Row(children: [
                Expanded(
                  child: GestureDetector(
                    onTap: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => const AiRoomSearchScreen(
                          hostelId: 0,
                          hostelName: 'StayBuddy',
                          city: 'Islamabad',
                          address: '',
                        ),
                      ),
                    ),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 11),
                      decoration: BoxDecoration(
                        color: AppTheme.tealMint.withOpacity(0.25),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                            color: AppTheme.tealMint.withOpacity(0.5),
                            width: 1),
                      ),
                      child: Row(children: [
                        const Icon(Icons.bedroom_parent_rounded,
                            color: Colors.white, size: 16),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text('AI Room Search',
                              style: GoogleFonts.dmSans(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w700,
                                  color: Colors.white)),
                        ),
                        const Icon(Icons.arrow_forward_ios_rounded,
                            color: Colors.white70, size: 12),
                      ]),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: GestureDetector(
                    onTap: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (_) => const AiRecommendationScreen()),
                    ),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 11),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                            color: Colors.white.withOpacity(0.3),
                            width: 1),
                      ),
                      child: Row(children: [
                        const Icon(Icons.auto_awesome_rounded,
                            color: Colors.white, size: 16),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text('AI Hostel Picks',
                              style: GoogleFonts.dmSans(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w700,
                                  color: Colors.white)),
                        ),
                        const Icon(Icons.arrow_forward_ios_rounded,
                            color: Colors.white70, size: 12),
                      ]),
                    ),
                  ),
                ),
              ]),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFilterRow() {
    return Container(
      color: AppTheme.bg,
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: SizedBox(
        height: 34,
        child: ListView.separated(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          itemCount: _filters.length,
          separatorBuilder: (_, __) => const SizedBox(width: 8),
          itemBuilder: (_, i) {
            final f = _filters[i];
            final isActive = f == activeFilter;
            return GestureDetector(
              onTap: () => setState(() => activeFilter = f),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 7),
                decoration: BoxDecoration(
                  color: isActive ? AppTheme.teal : AppTheme.card,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: isActive ? AppTheme.teal : AppTheme.border,
                    width: 0.5,
                  ),
                ),
                child: Text(
                  f,
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
                    color: isActive ? Colors.white : AppTheme.textSecondary,
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildAiBanner() {
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const AiRecommendationScreen()),
      ),
      child: Container(
        margin: const EdgeInsets.fromLTRB(16, 10, 16, 0),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF063E40), AppTheme.teal],
          ),
          borderRadius: BorderRadius.circular(14),
          boxShadow: [
            BoxShadow(
                color: AppTheme.teal.withOpacity(0.22),
                blurRadius: 12,
                offset: const Offset(0, 4)),
          ],
        ),
        child: Row(children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.15),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.auto_awesome_rounded,
                color: AppTheme.tealMint, size: 20),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Try AI-Powered Room Search',
                  style: GoogleFonts.dmSans(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: Colors.white)),
              Text('Hybrid ML · Personalised picks just for you',
                  style: GoogleFonts.dmSans(
                      fontSize: 11,
                      color: Colors.white.withOpacity(0.65))),
            ]),
          ),
          const SizedBox(width: 10),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.18),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(
                  color: Colors.white.withOpacity(0.3), width: 0.5),
            ),
            child: Text('Try Now',
                style: GoogleFonts.dmSans(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: Colors.white)),
          ),
        ]),
      ),
    );
  }

  Widget _buildEmpty() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.search_off_rounded,
            size: 56,
            color: AppTheme.textMuted.withOpacity(0.4),
          ),
          const SizedBox(height: 14),
          Text(
            'No hostels found',
            style: GoogleFonts.dmSans(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: AppTheme.textSecondary,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Try a different search or filter',
            style: GoogleFonts.dmSans(
              fontSize: 13,
              color: AppTheme.textMuted,
            ),
          ),
        ],
      ),
    );
  }
}

// Color palette for card accents
const _cardColors = [
  [Color(0xFF9FE1CB), Color(0xFF5DCAA5)],
  [Color(0xFFB5D4F4), Color(0xFF85B7EB)],
  [Color(0xFFFAC775), Color(0xFFEF9F27)],
  [Color(0xFFF4C0D1), Color(0xFFED93B1)],
];

class _HostelCard extends StatelessWidget {
  final dynamic hostel;
  final int index;
  final VoidCallback onTap;

  const _HostelCard({
    required this.hostel,
    required this.index,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final name = hostel['name'] ?? 'Hostel';
    final city = hostel['city'] ?? '';
    final avail = (hostel['available_capacity'] ?? 0) as int;
    final total = (hostel['total_capacity'] ?? 0) as int;
    final colors = _cardColors[index % _cardColors.length];

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: ThreeDContainer(
        borderRadius: 18,
        child: Material(
          color: AppTheme.card,
          borderRadius: BorderRadius.circular(18),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(18),
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppTheme.border, width: 0.5),
            ),
            child: Row(
              children: [
                // Image placeholder
                Container(
                  width: 88,
                  height: 100,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: colors,
                    ),
                    borderRadius: const BorderRadius.horizontal(
                      left: Radius.circular(18),
                    ),
                  ),
                  child: Icon(
                    Icons.business_rounded,
                    size: 34,
                    color: colors[1].withOpacity(0.5) == colors[1]
                        ? Colors.white.withOpacity(0.7)
                        : Colors.white.withOpacity(0.7),
                  ),
                ),

                // Content
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.all(14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          name,
                          style: GoogleFonts.dmSans(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: AppTheme.textPrimary,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 3),
                        Row(
                          children: [
                            Icon(
                              Icons.location_on_rounded,
                              size: 12,
                              color: AppTheme.textMuted,
                            ),
                            const SizedBox(width: 3),
                            Expanded(
                              child: Text(
                                city,
                                style: GoogleFonts.dmSans(
                                  fontSize: 12,
                                  color: AppTheme.textMuted,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        // Badges
                        Row(
                          children: [
                            _badge(
                              label: '$avail beds left',
                              bg: avail > 0
                                  ? const Color(0xFFEAF3DE)
                                  : const Color(0xFFFCEBEB),
                              text: avail > 0
                                  ? const Color(0xFF3B6D11)
                                  : const Color(0xFFA32D2D),
                            ),
                          ],
                        ),
                        const SizedBox(height: 6),
                        // Capacity bar
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            ClipRRect(
                              borderRadius: BorderRadius.circular(3),
                              child: LinearProgressIndicator(
                                value: total > 0 ? avail / total : 0,
                                backgroundColor: AppTheme.bgSecondary,
                                color: avail > 0 ? AppTheme.tealLight : Colors.red.shade300,
                                minHeight: 4,
                              ),
                            ),
                            const SizedBox(height: 3),
                            Text(
                              '$avail / $total available',
                              style: GoogleFonts.dmSans(
                                fontSize: 10,
                                color: AppTheme.textMuted,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),

                Padding(
                  padding: const EdgeInsets.only(right: 14),
                  child: Icon(
                    Icons.chevron_right_rounded,
                    color: AppTheme.textMuted,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    ),
    );
  }

  Widget _badge({
    required String label,
    required Color bg,
    required Color text,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        label,
        style: GoogleFonts.dmSans(
          fontSize: 10,
          fontWeight: FontWeight.w600,
          color: text,
        ),
      ),
    );
  }
}